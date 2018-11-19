import json
import numpy as np
import matplotlib.pyplot as plt

import click


def plot_trike(trike):
    p0, p1, p = trike['p0'], trike['p1'], trike['p']
    loaded, to_load = trike['loaded'], trike['to_load']

    plt.plot(*zip(p0, p1), 'b--', lw=0.5)

    plt.annotate('{}'.format(-to_load), p, color='black')
    # c = 'none' if to_load >= 0 else 'r'
    c = 'black'

    l, h = plt.xlim()
    s = (h - l) / 50 * (2 + np.sqrt(loaded))
    plt.scatter(*p, s=s, color='rgbmykw'[trike['id']], marker='^')


def plot_regions(regions):
    locations = [region['p'] for region in regions]
    x, y = zip(*locations)
    loads = [region['loaded'] for region in regions]
    capacities = [region['capacity'] for region in regions]

    plt.scatter(x=x, y=y, s=np.array(loads) * 5)
    for i in range(len(locations)):
        s = '{}/{}'.format(loads[i],
                           capacities[i])
        plt.annotate(s, locations[i])

    plt.xlim(np.min(x) - 200, np.max(x) + 200)
    plt.ylim(np.min(y) - 200, np.max(y) + 200)


def plot_frame(frame):
    plot_regions(frame['regions'])

    for trike in frame["trikes"]:
        plot_trike(trike)


@click.command()
@click.argument('path')
def plot(path):
    with open(path, 'r') as f:
        frames = json.load(f)
    for frame in frames[::2]:
        plt.cla()
        plot_frame(frame)
        plt.show(block=False)
        plt.pause(0.05)


if __name__ == "__main__":
    plot()
