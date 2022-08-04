# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 19:22:42 2020

@author: Tim
"""

import re
# import sys
import glob

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class Hiking:
    def __init__(self, path_to_files):
        gpx_files = glob.glob(path_to_files + "etappen/*.txt")
        gpx_files.sort()

        self.places = np.loadtxt(
            path_to_files+"orte.txt",
            delimiter='\t',
            dtype=([('etappe', int),
                    ('distance', float),
                    ('altitude', float),
                    ('sleep', bool),
                    ('name', '<U30')])
            )

        self.mountains = np.loadtxt(
            path_to_files+"gebirge.txt",
            dtype=([('stop', int),
                    ('height', int),
                    ('name', '<U20')])
            )

        distance = []
        altitude = []
        etappen_length = [0]
        etappen = [[[0]], [[0]]]

        for idx, file in enumerate(gpx_files):
            data = np.loadtxt(file).T

            if idx == 0:
                distance = np.append(distance, data[0])
                etappen[0].append(data[0])
            else:
                etappen[0].append(data[0] + distance[-1])
                distance = np.append(distance, data[0] + distance[-1])

            altitude = np.append(altitude, data[1])
            etappen_length = np.append(etappen_length, data[0][-1])
            etappen[1].append(data[1])

        self.distance  = distance
        self.altitude  = altitude
        self.n_entries = len(distance)
        self.etappen   = np.array(etappen).T
        self.etappen_length = etappen_length
        self.max_distance   = np.max(distance)
        self.max_altitude   = np.max(altitude)
        self.path_to_files  = path_to_files

    def prepare_plot(self):
        font = {'size' : 11}
        mpl.rc('font', **font)
        self.fig = plt.figure(figsize=(9,3))
        self.ax = plt.subplot(111)
        self.colors = [
            "#543005",
            "#8c510a",
            "#bf812d",
            "#dfc27d",
            "#35978f",
            "#01665e",
            "#003c30"
            ]

    def rote_etappe(self, etappe):
        plt.plot(
            self.etappen[etappe,0],
            self.etappen[etappe,1],
            color='C3',
            linewidth=1.2
            )
        plt.text(
            np.mean(self.etappen[5,0]),
            3000,
            r"e%d" % etappe,
            ha='center',
            color='C3',
            fontsize=8,
            fontweight='bold'
            )

    def ortsbeschriftungen(self):
        name_counter = 0
        for plc in self.places:
            if plc['sleep']:
                name_counter += 1
                text_pos   = self.max_altitude + 300
                spc        = np.linspace(0, self.distance[-1], 31)
                text_pos_x = spc[name_counter-1]
                text_rot   = 45
                text_va    = 'bottom'
            else:
                continue
            #     text_pos = -600
            #     text_rot = -45
            #     text_va = 'top'

            # words = re.findall('([A-Z][\u00C0-\u017Fa-z]*)', plc['name'])
            words = re.findall('(/&*)', plc['name'])
            if   len(words) == 1:
                self.print_text = words[0]
            elif len(words) == 2:
                self.print_text = f"{words[0]} {words[1]}"
            elif len(words) == 3:
                self.print_text = f"{words[0]} {words[1]} {words[2]}"
            elif len(words) == 4:
                self.print_text = f"{words[0]} {words[1]} {words[2]} {words[3]}"

            if (plc['name']=="Marienplatz") or (plc['name']=="Markusplatz"):
                weights = 'bold'
            else:
                weights = 'normal'

            dist_tot = self.etappen_length.cumsum()
            name_distance = dist_tot[plc['etappe']-1] + plc['distance']

            plt.vlines(
                name_distance,
                plc['altitude'],
                text_pos,
                linestyle='--',
                color='grey',
                linewidth=0.5
                )
            self.ax.arrow(
                name_distance,
                plc['altitude'],
                0,
                self.max_altitude+100-plc['altitude'],
                linestyle=':',
                color='grey',
                linewidth=0.5,
                clip_on=False
                )
            self.ax.arrow(
                text_pos_x,
                text_pos-50,
                name_distance-text_pos_x,
                self.max_altitude+100-(text_pos-50),
                linestyle=':',
                color='grey',
                linewidth=0.5,
                clip_on=False
                )
            plt.text(
                text_pos_x,
                text_pos,
                plc['name'],
                va=text_va,
                fontsize=7,
                rotation=text_rot,
                weight=weights
                )
            plt.plot(
                name_distance,
                plc['altitude'],
                color='black',
                marker='o',
                zorder=9,
                markersize=0.8,
                linestyle='None'
                )

    def gebirge(self):
        mountain_start = 0
        for idx, plc in enumerate(self.mountains):
            try:
                mountain_end = np.where(self.distance>plc['stop'])[0][0]
            except IndexError:
                mountain_end = len(self.distance)

            x_fill = self.distance[mountain_start:mountain_end]
            y_fill = self.altitude[mountain_start:mountain_end]

            plt.plot(
                x_fill,
                y_fill,
                color=self.colors[idx],
                linewidth=0.5
                )

            rge = np.arange(0, self.max_altitude, 500)
            for quant in rge:
                quant_arr = np.ones(len(x_fill))*quant
                plt.fill_between(
                    x_fill,
                    0,
                    np.minimum(quant_arr, y_fill),
                    color=self.colors[idx],
                    alpha=0.25,
                    linewidth=0
                    )
            mountain_start = mountain_end-1

    def gebirgebeschriftungen(self):
        mountain_start = 0
        for idx, plc in enumerate(self.mountains):
            try:
                mountain_end = np.where(self.distance>plc['stop'])[0][0]
            except IndexError:
                mountain_end = len(self.distance)

            x_fill = self.distance[mountain_start:mountain_end]

            words = re.findall('([A-Z][\u00C0-\u017Fa-z]*)', plc['name'])
            if len(words) == 1:
                print_name = words[0]
            else:
                print_name = f"{words[0]}\n{words[1]}"

            plt.text(
                np.mean(x_fill),
                plc['height'],
                print_name,
                ha='center',
                rotation=15-idx*5,
                fontsize=10
                )

            mountain_start = mountain_end-1

    def green_color_shading(self):
        for quant in np.arange(0, self.max_altitude, 500):
            quant_arr = np.ones(self.n_entries)*quant
            plt.fill_between(
                self.distance,
                0,
                np.minimum(quant_arr, self.altitude),
                color='C2',
                alpha=0.1,
                linewidth=0
                )

    def sealine(self):
        plt.plot(
            self.distance,
            np.zeros(self.n_entries),
            color='#7bccc4',
            linewidth=0.5
            )
        plt.fill_between(
            self.distance,
            -100,
            0,
            color='#7bccc4',
            alpha=0.5,
            linewidth=0
            )

    def finish_plot(self, file_name):
        # Hide the right and top spines
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)

        # further settings
        plt.xlabel("Entfernung (km)")
        plt.ylabel("Höhe (m)")
        plt.xlim(0, self.max_distance)
        plt.ylim(-100, self.max_altitude +100)
        plt.xticks(np.arange(0, self.max_distance, 50))
        if file_name:
            plt.savefig(
                self.path_to_files+file_name,
                dpi=1000,
                transparent=True,
                bbox_inches='tight'
                )
        plt.show()

    def statistics(self):
        slope = np.diff(self.altitude)
        signs = np.sign(slope)

        ascent = []
        descent = []

        sign_tmp = signs[0]
        base_tmp = self.altitude[0]
        upwards  = True
        for idx, step in enumerate(signs[1:]):
    #        print(idx, step, altitude[idx+1])
            if not step in [0, sign_tmp]:
                if upwards:
                    ascent.append(self.altitude[idx+1]-base_tmp)
                else:
                    descent.append(base_tmp-self.altitude[idx+1])
                sign_tmp = step
                base_tmp = self.altitude[idx]
                upwards  = not upwards
    #            print(upwards, ascent, descent)

        ascent  = np.sum(ascent)
        descent = np.sum(descent)
        error   = 521-(descent-ascent)

        print(f'ascent:  {ascent}\ndescent: {descent}')
        print(ascent-error)
        print(error)
        print(self.distance[-1])

if __name__ == '__main__':
    FILE_PATH = r"C:/Users/timbr/Documents/github/hoehenprofil/"
    hike = Hiking(FILE_PATH)

#%%
    hike.prepare_plot()
    # hike.rote_etappe(etappe=5)
    hike.ortsbeschriftungen()
    hike.gebirge()
    # hike.green_color_shading()
    # hike.gebirgebeschriftungen()
    hike.sealine()
    hike.finish_plot(file_name=None)

    hike.statistics()
