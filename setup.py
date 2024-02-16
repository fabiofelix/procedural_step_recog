from setuptools import setup, find_packages
setup(
    name='step_recog', version='1.0', packages=find_packages(), 
    install_requires=[
        'torch', 'ultralytics==8.0.239',
        'clip @ git+https://github.com/openai/CLIP.git@main#egg=clip',
    ]
)
