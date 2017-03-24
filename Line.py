import numpy as np
import matplotlib.pyplot as plt
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #array of polynomial coefficients of the last n iterations
        self.recent_nfits = [] #np.array([0,0,0], dtype='float')
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = []
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between avgerage and new fits
        self.avg_diffs = np.array([0,0,0], dtype='float')
        #get average of last n elements
        self.num_avg = 5
        #average of last n fits - 2 in our case
        self.avg_lastnfits = np.array([0,0,0], dtype='float')
        #difference in fit coefficients between last and new fits
        self.recent_diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    # def __init__(self, allx, ally):
        # self.allx = allx
        # self.ally = ally

    def isLineDetected(self, fit):
        total_fits = len(self.recent_nfits)
        # print("total_fits are:",total_fits)
        # print("all nfits shape is:",self.recent_nfits.shape)
        if total_fits == 1:
            # print("all nfits are:",self.recent_nfits)
            # print("fit is:",fit)
            # print("last nfits are:",self.recent_nfits[-1])
            self.avg_diffs = np.array([np.subtract(self.best_fit, fit)])
            self.recent_diffs = np.array([np.subtract(self.recent_nfits[-1], fit)])
            # print("best_fit is:",self.best_fit)
            # print("rdiffs is:",self.recent_diffs)
        elif total_fits > 1:
            # print("all nfits are:",self.recent_nfits)
            # print("fit is:",fit)
            # print("last nfits are:",self.recent_nfits[-1])
            adiff = np.subtract(self.best_fit, fit)
            rdiff = np.subtract(self.recent_nfits[-1], fit)
            # print("best_fit is:",self.best_fit)
            # print("diff is:",diff)
            self.avg_diffs = np.append(self.avg_diffs, [adiff], axis=0)
            self.recent_diffs = np.append(self.recent_diffs, [rdiff], axis=0)
            # print("appended rdiffs is:",self.recent_diffs)
            # print("all nfits is:",self.recent_nfits)
            # two_avg_fit = np.array([self.recent_nfits[-1]])
            # print("second last nfits is:",self.recent_nfits[-2])
            # print("last nfits is:",self.recent_nfits[-1])
            # print("two avg is:",two_avg_fit)
            # two_avg_fit = np.append(two_avg_fit, [self.recent_nfits[-2]], axis=0)
            if total_fits >= self.num_avg:
                last_navg_fit = self.recent_nfits[-self.num_avg:]
                num_avg_wgts = [1] * self.num_avg
                num_avg_wgts[0] = 2
            else:
                last_navg_fit = self.recent_nfits
                num_avg_wgts = [1] * total_fits
            # print("two avg is:",two_avg_fit)
            # two_avg_fit = np.average(two_avg_fit, weights=[1, 1], axis=0)
            # print("last_navg_fit is:",last_navg_fit)
            # print("num_avg_wgts is:",num_avg_wgts)
            last_navg_fit = np.average(last_navg_fit, weights=num_avg_wgts, axis=0)
            # print("averaged two avg is:",two_avg_fit)
            # print("averaged last_navg_fit is:",last_navg_fit)
            line_diff = np.subtract(last_navg_fit, fit)
            # print("line_diff is:",line_diff)
            # print("line_diff[0] is:",line_diff[0])
            # checking whether the polarity of new fit has changed
            # if (two_avg_fit[0]>0) != (fit[0]>0):
            #     print("flipped quadratic coeff is:", line_diff[0])
            #     print("flipped complete coeff is:", line_diff)
            #     print("fit is:",fit)
            #     print("two_avg_fit is:",two_avg_fit)
            # if (two_avg_fit[1]>0) != (fit[1]>0):
            #     print("flipped linear coeff is:", line_diff[1])
            #     print("flipped complete coeff is:", line_diff)
            #     print("fit is:",fit)
            #     print("two_avg_fit is:",two_avg_fit)
            # if (two_avg_fit[2]>0) != (fit[2]>0):
            #     print("flipped simple coeff is:", line_diff[2])
            #     print("flipped complete coeff is:", line_diff)
            #     print("fit is:",fit)
            #     print("two_avg_fit is:",two_avg_fit)
            # Figuring out the difference between the average of last n fits and the current fit
            # if the signs of quadratic and linear coefficients are reversed and
            # the absolute valute of quadratic coefficient of current fit is greater or equal to 3 decimal places
            # then we mark the line as NOT detected
            if (last_navg_fit[0]>0) != (fit[0]>0) and (last_navg_fit[1]>0) != (fit[1]>0) and (np.floor(np.log10(np.abs(fit[0]))) >= -3):
                print("flipped quadratic coeff is:", line_diff[0])
                print("flipped linear coeff is:", line_diff[1])
                print("flipped complete coeff is:", line_diff)
                print("fit is:",fit)
                print("last_navg_fit is:",last_navg_fit)
                self.avg_lastnfits = last_navg_fit

                return False


        return True

    def addFit(self, fit):
        self.current_fit = fit
        total_fits = len(self.recent_nfits)
        # print("total_fits:", total_fits)
        # print("fit is:",fit)
        # print("recent_nfits is:",self.recent_nfits)
        # print("best_fit is:",self.best_fit)
        if total_fits == 0:
            self.recent_nfits = np.array([fit])
            self.best_fit = np.array(fit)
        else:
            # weights = np.array([total_fits, 1], dtype='int')
            # self.best_fit = np.average(self.best_fit, weights)
            self.recent_nfits = np.append(self.recent_nfits, [fit], axis=0)
            two_fit = np.append([self.best_fit], [fit], axis=0)
            # print("b4 best_fit is:",self.best_fit)
            self.best_fit = np.average(two_fit, weights=[total_fits, 1], axis=0)
            # print("avg best_fit is:",self.best_fit)


    def printVals(self):
        print("recent_nfits is")
        print(self.recent_nfits)
        print("best_fit is")
        print(self.best_fit)
