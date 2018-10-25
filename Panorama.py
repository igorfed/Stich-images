import matplotlib.pyplot as plt

import Stiching as mf

def Stich (**args):

    val = list(args.values())

    __mf= mf.Stich([val[0], val[1]], val[2],newSize = val[3],  DetectorType = val[4])

    #mf.imshow('Source images', __mf.image0, __mf.image1, None)

    #plt.show()
    print ('done')

if __name__ == '__main__':
    args = mf.main()
    Stich(**vars(args))
