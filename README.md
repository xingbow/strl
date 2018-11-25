# Towards Better Bike Reposition

> Our [github repository](https://github.com/enhuiz/strl) .

## Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Download data

Three data file is used to train the model:

`./data/weather.csv`, which contain the weather data.

`./data/transition.csv`, processed bikes transition data.

`./data/stationStatus.csv`, station information.

If there are no data found in this folder, please download the data.zip file in [this github repository](https://github.com/enhuiz/Bike-Reposition-Data).

## Run Experiment

In the root directory, run:

```bash
python3 src/expt.py
```

## Run Artificial Experiment

The artificial experiment is designed for debugging our agent before the real data is ready (we want to develop the project in parallel), it is more fun and intuitive. In the root directory, run:

```bash
python3 src/artificial_expt.py
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

## Experiment Environment

- **Operating System**: Darwin-18.2.0-x86_64-i386-64bit
- **Python Version**: 3.6.5
- **CPU**: Intel(R) Core(TM) i5-7267U CPU @ 3.10GHz
