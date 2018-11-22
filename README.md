# Towards better bike reposition

> Our [github repository](https://github.com/enhuiz/strl) .

## Install Dependencies

```bash
pip3 install requirements.txt
```

## Download data

Three data file is used to train the model:

`./data/weather.csv`, which contain the weather data.

`./data/transition.csv`, processed bikes transition data.

`./data/stationStatus.csv`, station information.

If there are no data found in this folder, please download the data.zip file in [this github repository]( https://github.com/enhuiz/Bike-Reposition-Data).

## Run Experiment

In the root directory, run:

```bash
python3 src/expt.py -F results
```

## Visualization

```bash
python snapshots/visualize.py snapshots/json/file-to-visualize.json
```

## Directory Structure

```bash
├── README.md
├── cluster                           # clustering, a README file is inside it
├── data                              # data folder
├── requirements.txt
├── results                           # the experiment result
├── snapshots
│   └── visualize.py                  # visualize the running snapshots (json file generated during running)
└── src
    ├── agent.py                      # all agents used
    ├── artificial_expt.py            # run experiment on artificial situation (debug)
    ├── artificial_simulator.py       # the artificial simulator (debug)
    ├── contrib
    │   ├── OIModel.py                # the OModel and the IModel
    │   ├── __init__.py
    │   └── core.py                   # interfaces
    ├── env.py                        # environment
    ├── expt.py                       # main experiment
    └── simulator.py                  # simulator
```
