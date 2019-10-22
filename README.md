# DeepGamer
This  project  presents  a  deep  learning  model  able  to  learn  how  toperform  several  tasks  using  the  1993  game  DOOM  as  environment.The agent is trained using raw pixels from the game screen and uses a deep learning variant of the Q-learning algorithm.  Several optimizations  techniques  were  applied  in  order  to  maximize  performance  and results.

## Clone the repo
You can clone this repo via https using the following command:

`git clone --recursive https://github.com/melchor629/retro`

## Run this project
### Get dependencies
The only major dependencies that you have to manually install are used for [VizDoom](http://vizdoom.cs.put.edu.pl/). Follow their [guide](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#deps) on how to install all the required dependencies.
### Set up environment
The best way to get all needed dependencies is to use [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). To install and initialize the environment, run the following commands:

```
conda env create -f src/environment.yml
conda activate deep-gamer
```

### Run the agent
There are two scenarios to train the agent. To run them, go into the `src` folder and run:
```
python basic.py
```
or
```
python defend_the_center.py
``
