import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    array = [[13,1,1,0,2,0],
         [3,9,6,0,1,0],
         [0,0,16,2,0,0],
         [0,0,0,13,0,0],
         [0,0,0,0,15,0],
         [0,0,1,0,0,15]]

    df_cm = pd.DataFrame(array, range(6), range(6))

    def displaymatrix(df_cm):


        sn.set(font_scale=1.4) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

        plt.show()

    displaymatrix(df_cm)