webui for traiNNer-redux

features:

- manage experiments
- view tensorboard graphs
- edit configs
- val image viewer

install:

- clone repo

- point launch.sh/launch.bat vars to traiNNer dir and python binary

- python -m pip install -r ./webui/requirements.txt   (traiNNer dependencies apply as well)

- run launch script

- alternatively run launch script --online arg for public cloudflare address

warning/info:

You might want to backup your experiments/configs before pointing this software directly at your traiNNer dir. I added rename and copy features and while I'm fairly confident it won't delete things, on the off-chance your files get nuked because of a bug I'm not responsible. 

The ui will attempt to keep experiment names, the config name field, and config filename matching. I tried to make it so this doesn't 100% matter so it can import old experiments but just so you know you should avoid manually changing the config name field and config filename and use the duplicate/rename experiment features instead. 

credits:

claude

traiNNer-redux
