import os
import pandas as pd
import numpy as np

PATH_TO_DATA = '/scratch/work/ptg/BBN/skills'
SKILLS = ['M5_X-Stat']
EK100_columns = ['narration_id','participant_id','video_id','narration_timestamp','start_timestamp','stop_timestamp','start_frame','stop_frame','narration','verb','verb_class','noun','noun_class','all_nouns','all_noun_classes']

for skill in SKILLS:

    path2data = os.path.join(PATH_TO_DATA,skill,'Data')
    vids = os.listdir(path2data)
    
    # random split into validation
    rand_idx = np.random.choice(len(vids),len(vids),replace=False)
    train_vids = [vids[i] for i in rand_idx[int(len(vids)*0.15):]]
    validation_vids = [vids[i] for i in rand_idx[:int(len(vids)*0.15)]]

    all_narrations = []
    for video_id in sorted(train_vids):

        video_files = os.listdir(os.path.join(path2data,video_id))
        #if f'{video_id}.action_labels_by_frame.txt' not in video_files:
        if f'{video_id}.skill_labels_by_frame.txt' not in video_files:
            print(f'{video_id} missing skill labels. Skipping video.')
            continue

        lab_df = pd.read_csv(os.path.join(path2data,video_id,f'{video_id}.skill_labels_by_frame.txt'),sep='\t',names= ['start_frame','stop_frame','narration'])
        all_narrations.extend(list(lab_df.narration))
    STEPS = {k:i for i, k in enumerate(list(set(all_narrations)))}
    print('steps indexed in metadata:')
    for k,v in STEPS.items():
        print(v,k)

    for j,vids in enumerate([train_vids, validation_vids]):

        narration_ids = []
        video_ids = []
        start_frames = []
        stop_frames = []
        narrations = []
        verbs = []
        verb_classes = []
        nouns = []
        noun_classes = []


        for video_id in sorted(vids):

            print(video_id)

            video_files = os.listdir(os.path.join(path2data,video_id))
            #if f'{video_id}.action_labels_by_frame.txt' not in video_files:
            if f'{video_id}.skill_labels_by_frame.txt' not in video_files:
                print(f'{video_id} missing skill labels. Skipping video.')
                continue

            lab_df = pd.read_csv(os.path.join(path2data,video_id,f'{video_id}.skill_labels_by_frame.txt'),sep='\t',names= ['start_frame','stop_frame','narration'])
            #verbs.extend([ACTION_verbs[n.split(' ')[1]]['key'] for n in lab_df.narration])
            #verb_classes.extend([ACTION_verbs[n.split(' ')[1]]['id'] for n in lab_df.narration])
            #nouns.extend([ACTION_nouns[n.split(' ')[-1]]['key'] if len(n.split(' '))>2 else 'none' for n in lab_df.narration])
            #noun_classes.extend([ACTION_nouns[n.split(' ')[-1]]['id'] if len(n.split(' '))>2 else '0' for n in lab_df.narration])
            verbs.extend([STEPS[n] for n in lab_df.narration])
            verb_classes.extend([STEPS[n] for n in lab_df.narration])
            nouns.extend([STEPS[n] for n in lab_df.narration])
            noun_classes.extend([STEPS[n] for n in lab_df.narration])

            narration_ids.extend([f'{video_id}_{i}' for i in range(len(lab_df))])
            video_ids.extend([video_id]*len(lab_df))
            start_frames.extend(lab_df.start_frame)
            stop_frames.extend(lab_df.stop_frame)
            narrations.extend(lab_df.narration)

        df = pd.DataFrame(
            {'narration_id'   :narration_ids,
             'video_id'       :video_ids,
             'start_frame'    :start_frames,
             'stop_frame'     :stop_frames,
             'narration'      :narrations,
             'verb'           :verbs,
             'verb_class'    :verb_classes,
             'noun'           :nouns,
             'noun_class'    :noun_classes}
        )
        df = df.set_index('narration_id')
        df.to_csv('csv/BBN_train.csv' if j==0 else 'csv/BBN_validation.csv')

        print(df)
