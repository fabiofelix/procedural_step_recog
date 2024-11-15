'''This is a test of perception utilizing the model server

'''
import os
import time
from typing import List
import asyncio
import orjson
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np
import torch

import ray
import functools
import ptgctl
import ptgctl.async_graph as ag
from ptgctl import holoframe
from models import AllInOneModel

ray.init(num_gpus=2)


model = AllInOneModel.remote(model_type=[0])
# model2 = AllInOneModel.remote(model_type=[1])

MODELS = {
    None: model,
    # 'model2': model2,
}
if torch.cuda.device_count() > 1:
    MODELS['model2'] = AllInOneModel.remote(model_type=[1])

holoframe_load = functools.lru_cache(maxsize=32)(holoframe.load)

class RecipeExit(Exception):
    pass

class Perception:
    def __init__(self, **kw):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'perception',
                              password=os.getenv('API_PASS') or 'perception')

    @torch.no_grad()
    @ptgctl.util.async2sync
    async def run(self, *a, **kw):
        '''Persistent running app, with error handling and recipe status watching.'''
        self.session = MultiStepsSession(MODELS)
        while True:
            try:
                await self.run_loop(*a, **kw)
            except RecipeExit as e:
                print(e)
                await asyncio.sleep(5)
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    async def run_loop(self):
        # stream ids
        with logging_redirect_tqdm():
            async with ag.Graph() as g:
                q_rgb, = g.add_producer(self.reader, g.add_queue(ag.Queue))
                q_proc = g.add_consumer(self.processor, q_rgb, g.add_queue(ag.Queue))
                g.add_consumer(self.writer, q_proc)
            print("finished")

    async def reader(self, queue, prefix=None):
        # stream ids
        in_sid = f'{prefix or ""}main'
        recipe_sid = f'{prefix or ""}event:recipe:id'
        vocab_sid = f'{prefix or ""}event:recipes'

        t0 = time.time()
        async with self.api.data_pull_connect([in_sid, recipe_sid, vocab_sid], ack=True, latest=False) as ws_pull:
            recipe_id = self.api.session.current_recipe()
            if recipe_id is not None:
                print('recipe id', recipe_id)
                await self.session.load_skill(recipe_id)
            pbar = tqdm.tqdm()
            while True:
                pbar.set_description('read: waiting for data...')
                for sid, t, d in await ws_pull.recv_data():
                    pbar.set_description(f'read: {sid} {t}')
                    pbar.update()
                    try:
                        # watch recipe changes
                        if sid == recipe_sid or sid == vocab_sid:
                            d = d.decode() if d else None
                            print("recipe changed", recipe_id, '->', d, flush=True)
                            recipe_id = d if d else None
                            if recipe_id:
                                print('recipe id', recipe_id)
                                await self.session.load_skill(recipe_id)
                            continue
                        if recipe_id is None:
                            continue
                        queue.push([sid, t, holoframe_load(d)['image']])
                        await asyncio.sleep(1e-2)
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        await asyncio.sleep(1e-1)


    async def processor(self, queue, out_queue):
        pbar = tqdm.tqdm()
        while True:
            pbar.set_description('processor waiting for data...')
            sid, t, img = await queue.get()
            try:
                pbar.set_description(f'processor got {sid} {t}')
                pbar.update()
                preds = await self.session.on_image([img])
                if preds is not None:
                        out_queue.push([preds, t])
            except Exception as e:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1e-1)
            finally:
                queue.task_done()
        print("processor done")

    async def writer(self, queue):
        '''Run the recipe.'''
        async with self.api.data_push_connect(self.session.out_sids, batch=True) as ws_push:
            # pbar = tqdm.tqdm()
            while True:
                # pbar.set_description('writer waiting for data...')
                preds, timestamp = await queue.get()
                try:
                    # pbar.set_description(f'writer got {set(preds)} {timestamp}')
                    # pbar.update()
                    await ws_write_data_dict(ws_push, preds, timestamp)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(1e-1)
                finally:
                    queue.task_done()
        print("writer done")


def dump_data_dict(data, timestamp):
    data = {k: v for k, v in data.items() if v is not None}
    stream_ids = list(data)
    json_data = [jsondump(x) for x in data.values()]
    timestamps = [noconflict_ts(timestamp)]*len(data)
    return stream_ids, json_data, timestamps
    
async def ws_write_data_dict(ws_push, data, timestamp):
    data = {k: v for k, v in data.items() if v is not None}
    await ws_push.send_data(
        [jsondump(x) for x in data.values()], 
        list(data.keys()), 
        [noconflict_ts(timestamp)]*len(data))

def noconflict_ts(ts):
    return ts.split('-')[0] + '-*'

def jsondump(data):
    return orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)


class MultiStepsSession:
    def __init__(self, models):
        self.sessions = []
        for prefix, model in models.items():
            self.sessions.append(StepsSession(prefix, model))
        s = self.sessions[0]
        self.out_sids = s.out_sids

    async def load_skill(self, recipe_id):
        await asyncio.gather(*[s.load_skill(recipe_id) for s in self.sessions])

    async def on_image(self, imgs):
        data = {}
        ds = await asyncio.gather(*[s.on_image(imgs) for s in self.sessions])
        for d in ds:
            data.update(d or {})
        return data
ag.Queue.push = ag.Queue.put_nowait


class StepsSession:
    #MAX_RATE_SECS = 0.5 #0.3
    MAX_RATE_SECS = 1.0
    def __init__(self, prefix=None, model=model):
        # models
        self.model = model

        # output names
        prefix = prefix or ""
        self.step_sid = f'{prefix}omnimix:step'
        self.step_id_sid = f'{prefix}omnimix:step:id'
        self.box_sid = f'{prefix}detic:image'
        self.steps_sid = f'{prefix}omnimix:steps:sm'
        self.steps_raw_sid = f'{prefix}omnimix:steps:sm:raw'
        self.out_sids = [
            self.step_sid,
            self.step_id_sid,
            self.box_sid,
            self.steps_sid,
            self.steps_raw_sid,
        ]
        print(prefix, self.out_sids)
        self.t0 = None

    async def load_skill(self, recipe_id):
        print(recipe_id)
        await self.model.load_skill.remote(recipe_id)
        vocab = await self.model.get_steps.remote()

        self.vocab = np.concatenate([np.asarray([f'{i+1}|{v}' for i, v in enumerate(vocab)]), np.array(['|OTHER'])])
        self.vocab_list = self.vocab.tolist()

        id_vocab = np.concatenate([np.arange(len(vocab)), np.array([-1])])
        self.id_vocab_list = id_vocab.astype(str).tolist()

        self.t0 = None
        self.local_buffer = []

    async def on_image(self, imgs: List[np.ndarray]):
        self.local_buffer.extend(imgs)

        t = time.time()
        if self.t0 is None:
            self.t0 = 0
        if self.MAX_RATE_SECS > (t - self.t0) + 0.03:
            return {}
        self.t0 = t

        imgs, self.local_buffer = self.local_buffer, []
        steps, objects, state = await self.model.forward.remote(imgs)
        if steps is None:
            return {}
        # convert everything to json-friendly
        step_list = steps[0].detach().cpu().tolist()
        return {
            self.step_sid: dict(zip(self.vocab_list, step_list)),
            self.step_id_sid: dict(zip(self.id_vocab_list, step_list)),
            self.box_sid: objects,#as_v1_objs(xyxyn, confs, class_ids, labels),
            self.steps_sid: {
                "all_steps": self.get_steps(state, step_list),
                "error_status": False,
                "error_description": "",
            },
            self.steps_raw_sid: state.tolist(),
        }

    def get_steps(self, state, confidence):
        return [
            {
                'number': i+1, 
                'name': f'{i+1}', # self.vocab[i], 
                'state': STATES[state[i]], 
                'confidence': confidence[i] if state[i]==1 else 1
            }
            for i in range(len(state))
        ]

STATES = ['unobserved', 'current', 'done']


if __name__ == '__main__':
    import fire
    fire.Fire(Perception)
