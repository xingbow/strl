import json
import numpy as np
import matplotlib.pyplot as plt
import click
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
plt.style.use('ggplot')
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'


COLORS = 'rgbmyk' * 10


def plot_trike(trike):
    p0, p1, p = trike['p0'], trike['p1'], trike['p']
    loaded, to_load, capacity = trike['loaded'], trike['to_load'], trike['capacity']

    plt.plot(*zip(p0, p1), 'b--', lw=0.5)

    plt.annotate('   {}'.format(loaded), p, color='black', fontsize=10)

    color = COLORS[trike['id']]

    # s = (size) / 100
    # plt.scatter(*p, s=s, color=color, marker='^')

    canvas_width = np.diff(plt.xlim())
    canvas_height = np.diff(plt.ylim())

    width = canvas_width * 2 * 1e-2
    height = canvas_height * 8 * 1e-2

    plt.gca().add_patch(patches.Rectangle(
        p, width, height * (loaded / capacity), linewidth=1,
        edgecolor='black',
        facecolor='g'))
    plt.gca().add_patch(patches.Rectangle(
        p, width,
        height * ((to_load + loaded) / capacity),
        linewidth=1,
        edgecolor='r',
        facecolor='none'))
    plt.gca().add_patch(patches.Rectangle(
        p, width, height, linewidth=1,
        edgecolor='black',
        facecolor='none'))


def plot_regions(regions):
    locations = [region['p'] for region in regions]
    x, y = zip(*locations)
    loads = [region['loaded'] for region in regions]
    capacities = [region['capacity'] for region in regions]

    plt.scatter(x=x, y=y, s=np.array(loads) * 2)
    for i in range(len(locations)):
        s = '{}/{} (R{})'.format(loads[i],
                                 capacities[i],
                                 regions[i]['id'])
        plt.annotate(s, locations[i], fontsize=10)

    plt.xlim(np.min(x) - 400, np.max(x) + 400)
    plt.ylim(np.min(y) - 500, np.max(y) + 500)


def plot_frame(frame):
    plt.cla()
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plot_regions(frame['regions'])
    for trike in frame["trikes"]:
        plot_trike(trike)


@click.command()
@click.argument('path')
def plot(path):
    with open(path, 'r') as f:
        frames = json.load(f)
    plt.tight_layout(True)
    anim = FuncAnimation(fig=plt.gcf(),
                         func=lambda i: print(i) or plot_frame(frames[i]),
                         frames=len(frames))

    # plt.show()
    anim.save('animate.gif', dpi=100, writer='imagemagick')


if __name__ == "__main__":
    plot()


def RL(env, agent):
    while True:
        state = env.observe()
        action = agent.act(state)
        reward = env.update(action)
        agent.learn(reward)
