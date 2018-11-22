"""
Example of animation
"""
from geoplotlib.layers import BaseLayer
from geoplotlib.core import BatchPainter
import geoplotlib
from geoplotlib.colors import colorbrewer
from geoplotlib.utils import epoch_to_str, BoundingBox, read_csv


class TrailsLayer(BaseLayer):

    def __init__(self):
        self.data = read_csv('trike_route.csv')
        self.cmap = colorbrewer(self.data['id'], alpha=220)
        self.t = self.data['timestamp'].min()
        self.painter = BatchPainter()


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter = BatchPainter()
        df = self.data.where((self.data['timestamp'] > self.t) & (self.data['timestamp'] <= self.t + 5*60))

        for tid in set(df['id']):
            grp = df.where(df['id'] == tid)
            self.painter.set_color(self.cmap[tid])
            x, y = proj.lonlat_to_screen(grp['lon'], grp['lat'])
            #self.painter.points(x, y, 15)
            for j in range(len(grp)):
                if grp['id'][j]>=100:
                    self.painter.circle_filled(x[j], y[j],10)
                else:
                    self.painter.points(x[j], y[j], 30)
        self.t += 30

        if self.t > self.data['timestamp'].max():
            self.t = self.data['timestamp'].min()

        self.painter.batch_draw()
        ui_manager.info(epoch_to_str(self.t))


    def bbox(self):
        return BoundingBox(north=40.78, west=-74.1, south=40.67, east=-73.9)


geoplotlib.add_layer(TrailsLayer())
geoplotlib.show()
