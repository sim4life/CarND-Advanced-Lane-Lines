import numpy as np
import matplotlib.pyplot as plt
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    # def __init__(self, allx, ally):
        # self.allx = allx
        # self.ally = ally

    def addFit(self, fit):
        self.current_fit = fit
        self.recent_xfitted.append(fit)
        total_fits = len(self.recent_xfitted)
        if total_fits == 1:
            self.best_fit = fit
        # else:
            # weights = np.array([total_fits, 1], dtype='int')
            # self.best_fit = np.average(self.best_fit, weights)


    def printVals(self):

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        f.tight_layout()

        # ax1.imshow(binary_warped, cmap='gray')
        ax1.hist(self.allx)
        ax1.set_title('AllX', fontsize=8)
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Frequency")

        ax2.hist(self.ally)
        ax2.set_title('AllY', fontsize=8)
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Frequency")

        plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.1)
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
