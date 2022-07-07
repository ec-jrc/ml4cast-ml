import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

def dbl_logistic_model ( p, x ):
    """A double logistic model, as in Sobrino and Juliean, or Zhang et al, see https://www.uv.es/juy/Doc/Julien-Sobrino_GIMMS-phenology_2009_IJRS.pdf"""
    #p[0] is base NDVI
    #p[1] is maxNDVI - p[0]
    #p[2] is dU, rate of increase at green-up inflexion point
    #p[3] is U, the position of the inflexion point during green-up
    #p[4] is dD, , rate of decrease at senescence inflexion point A
    #p[5] is D, the position of the inflexion point during decay

    return p[0] + p[1] * (1./(1+np.exp(-p[2] *  (x-p[3]))) +
                          1./(1+np.exp(p[4] * (x-p[5]))) - 1 )

def mismatch_function ( p, pheno_func, ndvi, days_from0):
    """"doys_from0 is the x, in days, and starts with 0"""
    fitness = lambda p, ndvi, days_from0,: \
        ndvi - pheno_func(p, days_from0)
    oot = fitness(p, ndvi, days_from0)
    return oot



def fit_phenology_model (pheno_model, sndvi0, days0):
    """Fit a double logistic to input data"""
    if pheno_model == "dbl_logistic":
        n_params = 6
        pheno_func = dbl_logistic_model
    xinit = [.5, ] * n_params
    # Dbl_logistic might require sensible starting point
    if pheno_model == "dbl_logistic":
        xinit[0] = sndvi0.min()
        xinit[1] = sndvi0.max() - sndvi0.min()
        xinit[2] = 0.19
        xinit[4] = 0.13
        # first approximation of inflexion points (half way between min and max)
        itom = np.argmax(sndvi0)             # locate the t of max
        ilmin = np.argmin(sndvi0[:itom])     # now locate the t of the left min
        irmin = itom + np.argmin(sndvi0[itom:])     # now locate the t of the right min
        xinit[3] = days0[ilmin]
        xinit[5] = days0[irmin]
        print('xinit')
        print(xinit)
        (xsol, msg) = leastsq(mismatch_function, xinit, args=(pheno_func, sndvi0, days0), maxfev=1000000) #https://github.com/jgomezdans/phenology/blob/master/pheno/phenology.py line 123

    return (xsol, msg)

def compute_pheno_3timings (x, yfit, prctSOS, prctSEN, prctEOS):
    """It compute pheno timings based on a fitted function at daily time step"""
    #dict to store results
    pheno = dict()
    #locate max
    iMax = np.argmax(yfit)
    pheno['TOM'] = x[iMax]
    pheno['Ymax'] = yfit[iMax]
    #define left minimum level and amplitude
    iLeftMin = np.argmin(yfit[:iMax])
    LeftMin = yfit[iLeftMin]
    LeftAmp = pheno['Ymax'] - LeftMin
    #define right minimum level and amplitude
    iRightMin = np.argmin(yfit[iMax:])
    RightMin = yfit[iRightMin+iMax]
    RightAmp = pheno['Ymax'] - RightMin
    #locate when, during green up, yfit exceeds base lft value plus prctSOS of green-up amplitude
    iSOS = np.argwhere(yfit[:iMax]-(LeftMin + LeftAmp * float(prctSOS)/100.0)>=0)[0]
    pheno['SOS'] = x[iSOS[0]]
    # locate when, during decay, yfit drops below base right value plus prctSOS of decay amplitude
    iEOS = iMax + np.argwhere(yfit[iMax:] - (RightMin + RightAmp * float(prctEOS)/100.0) <=0)[0]
    pheno['EOS'] = x[iEOS[0]]
    # locate when, during decay, yfit drops below base right value plus prctSEN of decay amplitude
    iSEN = iMax + np.argwhere(yfit[iMax:] - (RightMin + RightAmp * float(prctSEN) / 100.0) <= 0)[0]
    pheno['SEN'] = x[iSEN[0]]
    return pheno



def pheno_lta_dek_mono_wrapper_v2(ndvi, dek, prctSOS, prctSEN, prctEOS, startRangeDekFrom0=None, endRangeDekFrom0=None):
    """manage the pheno fit. ndvi and doy must be arrays.
       handles the fit for monomodal on lta"""
    # startRangeDekFrom0 and endRangeDekFrom0 are used in the case there is a bimodal season to define the range for the fit

    # smooth ndvi
    # smooth the values (running mean of width 3)

    ndvi3 = np.concatenate((ndvi, ndvi, ndvi))
    dek3 = np.concatenate((dek, dek, dek))
    sndvi3 = np.convolve(ndvi3, np.ones(3) / 3.0, mode='same')
    if startRangeDekFrom0 is None:
        #one season covering the whole year
        mono = True
        startRangeDekFrom0 = 0
        endRangeDekFrom0 = 35
    else:
        mono = False
        startRangeDekFrom0 = int(startRangeDekFrom0)
        endRangeDekFrom0 = int(endRangeDekFrom0)
    # cut out the time domain of interest, taking into account that startRangeDekFrom0 can be > than endRangeDekFrom0
    if endRangeDekFrom0 > startRangeDekFrom0:
        lengthOfPeriod =  endRangeDekFrom0 - startRangeDekFrom0 + 1
    else:
        lengthOfPeriod = 36 - startRangeDekFrom0 + endRangeDekFrom0 + 1
    # first year has indices 0-35, second 36-71, third 72-107
    # always take data from central year, well smoothed
    sndvi = sndvi3[startRangeDekFrom0+36:startRangeDekFrom0+36+lengthOfPeriod]  #note: elemets up to startRangeDekFrom0+lengthOfPeriod-1 are taken
    dek = dek3[startRangeDekFrom0+36:startRangeDekFrom0+36+lengthOfPeriod]
    # Map the time profile so that it starts at time of minimum (align to solar year or prescribed period)
    # get position of the min, consider as day 0 and re-arrange vectors
    dek_from0 = dek * 0
    if mono == True:
        iMin =np.argmin(np.array(ndvi))
        dek_from0[iMin:] = dek[iMin:] - dek[iMin]
        dek_from0[:iMin] = dek[:iMin] + (36 - dek[-1]) + dek_from0[-1]
        isort = np.argsort(dek_from0)
        sndvi_from_day0 = sndvi[isort]
        dek_from0 = dek_from0[isort]
    else:
        iMin = 0
        dek_from0 = np.arange(lengthOfPeriod)
        sndvi_from_day0 = sndvi



    (xsol, msg) = fit_phenology_model("dbl_logistic", sndvi_from_day0, dek_from0)
    #print('xsol, msg')
    #print(xsol, msg)
    if msg != 1:
        print(msg)
    #reconstruct the forward model
    x = np.arange(0,36)
    yfit = dbl_logistic_model(xsol, x)
    #compute pheno paramters
    #pheno = compute_pheno_timings(x, yfit, prctSOS, prctEOS)
    pheno  = compute_pheno_3timings(x, yfit, prctSOS, prctSEN, prctEOS)
    # now remap days from 0 to actual deks

    for t in ('SOS', 'SEN', 'EOS', 'TOM'):
        pheno[t] = dek[iMin] + pheno[t]

    if False:
        plt.figure()
        plt.plot(day_from0, ndvi_from_day0, color='red', marker='+', linestyle='')#, linewidth=0.5, label=uniqueWilayaNames[i])
        plt.plot(x, yfit, color='blue', marker='', linestyle='-')
        axes = plt.gca()
        yrange = axes.get_ylim()
        plt.plot([pheno['TOM'],pheno['TOM']],[yrange[0],yfit[pheno['TOM']]], color='black', marker='', linestyle='--')
        plt.plot([pheno['SOS'], pheno['SOS']], [yrange[0], yfit[pheno['SOS']]], color='black', marker='', linestyle='--')
        plt.plot([pheno['SEN'], pheno['SEN']], [yrange[0], yfit[pheno['SEN']]], color='black', marker='', linestyle='--')
        plt.plot([pheno['EOS'], pheno['EOS']], [yrange[0], yfit[pheno['EOS']]], color='black', marker='', linestyle='--')
        axes.set_ylim(yrange)
        plt.xlabel('days from ' + str(doy[iMin]))
        plt.ylabel('NDVI')
        plt.show()
    #plt.close()


    return pheno


# OLD functions
# def compute_pheno_timings (x, yfit, prctSOS, prctEOS):
#     """It compute pheno timings based on a fitted function at daily time step"""
#     #dict to store results
#     pheno = dict()
#     #locate max
#     iMax = np.argmax(yfit)
#     pheno['TOM'] = x[iMax]
#     pheno['Ymax'] = yfit[iMax]
#     #define left minimum level and amplitude
#     iLeftMin = np.argmin(yfit[:iMax])
#     LeftMin = yfit[iLeftMin]
#     LeftAmp = pheno['Ymax'] - LeftMin
#     #define right minimum level and amplitude
#     iRightMin = np.argmin(yfit[iMax:])
#     RightMin = yfit[iRightMin+iMax]
#     RightAmp = pheno['Ymax'] - RightMin
#     #locate when, during green up, yfit exceeds base lft value plus prctSOS of green-up amplitude
#     iSOS = np.argwhere(yfit[:iMax]-(LeftMin + LeftAmp * float(prctSOS)/100.0)>=0)[0]
#     pheno['SOS'] = x[iSOS[0]]
#     # locate when, during decay, yfit drops below base right value plus prctSOS of decay amplitude
#     iEOS = iMax + np.argwhere(yfit[iMax:] - (RightMin + RightAmp * float(prctEOS)/100.0) <=0)[0]
#     pheno['EOS'] = x[iEOS[0]]
#     return pheno
# def pheno_lta_mono_wrapper (ndvi, doy, prctSOS, prctEOS):
#     """manage the pheno fit. ndvi and doy must be arrays.
#        handles the fit for monomodal on lta"""
#
#     # Map the time profile so that it starts at time of minimum (align to solar year)
#     # concatenate trice to avoide edges effect
#     # smooth the values (running mean of width 3)
#
#     ndLen = len(ndvi)
#     sndvi = np.convolve(np.concatenate((ndvi,ndvi,ndvi)), np.ones(3)/3.0, mode = 'same')
#     sndvi = sndvi[ndLen:ndLen*2]
#     # get position of the min, consider as day 0 and re-arrange vectors
#     iMin =np.argmin(np.array(sndvi))
#     day_from0 = doy * 0
#     day_from0[iMin:] = doy[iMin:] - doy[iMin]
#     day_from0[:iMin] = doy[:iMin] + (365 - doy[-1]) + day_from0[-1]
#     isort = np.argsort(day_from0)
#     ndvi_from_day0 = ndvi[isort]
#     day_from0 = day_from0[isort]
#     (xsol, msg) = fit_phenology_model("dbl_logistic", ndvi_from_day0, day_from0)
#     print('xsol, msg')
#     print(xsol, msg)
#     if msg != 1:
#         print(msg)
#     #reconstruct the forward model
#     x = np.arange(1, 366)
#     yfit = dbl_logistic_model(xsol, x)
#     #compute pheno paramters
#     pheno = compute_pheno_timings(x, yfit, prctSOS, prctEOS)
#     if False:
#         plt.figure()
#         plt.plot(day_from0, ndvi_from_day0, color='red', marker='+', linestyle='')#, linewidth=0.5, label=uniqueWilayaNames[i])
#         plt.plot(x, yfit, color='blue', marker='', linestyle='-')
#         axes = plt.gca()
#         yrange = axes.get_ylim()
#         plt.plot([pheno['TOM'],pheno['TOM']],[yrange[0],yfit[pheno['TOM']]], color='black', marker='', linestyle='--')
#         plt.plot([pheno['SOS'], pheno['SOS']], [yrange[0], yfit[pheno['SOS']]], color='black', marker='', linestyle='--')
#         plt.plot([pheno['EOS'], pheno['EOS']], [yrange[0], yfit[pheno['EOS']]], color='black', marker='', linestyle='--')
#         axes.set_ylim(yrange)
#         plt.xlabel('days from ' + str(doy[iMin]))
#         plt.ylabel('NDVI')
#         plt.show()
#     #plt.close()
#     #now remap days from 0 to actual doys
#     for t in ('SOS','EOS','TOM'):
#         pheno[t] = doy[iMin] + pheno[t]
#
#     return pheno
#
# def pheno_lta_dek_mono_wrapper (ndvi, dek, prctSOS, prctSEN, prctEOS):
#     """manage the pheno fit. ndvi and doy must be arrays.
#        handles the fit for monomodal on lta"""
#
#     # Map the time profile so that it starts at time of minimum (align to solar year)
#      # get position of the min, consider as day 0 and re-arrange vectors
#     iMin =np.argmin(np.array(ndvi))
#     dek_from0 = dek * 0
#     dek_from0[iMin:] = dek[iMin:] - dek[iMin]
#     dek_from0[:iMin] = dek[:iMin] + (36 - dek[-1]) + dek_from0[-1]
#     isort = np.argsort(dek_from0)
#     ndvi_from_day0 = ndvi[isort]
#     dek_from0 = dek_from0[isort]
#     (xsol, msg) = fit_phenology_model("dbl_logistic", ndvi_from_day0, dek_from0)
#     print('xsol, msg')
#     print(xsol, msg)
#     if msg != 1:
#         print(msg)
#     #reconstruct the forward model
#     x = np.arange(0,36)
#     yfit = dbl_logistic_model(xsol, x)
#     #compute pheno paramters
#     pheno = compute_pheno_timings(x, yfit, prctSOS, prctEOS)
#     pheno  = compute_pheno_3timings(x, yfit, prctSOS, prctSEN, prctEOS)
#     if False:
#         plt.figure()
#         plt.plot(day_from0, ndvi_from_day0, color='red', marker='+', linestyle='')#, linewidth=0.5, label=uniqueWilayaNames[i])
#         plt.plot(x, yfit, color='blue', marker='', linestyle='-')
#         axes = plt.gca()
#         yrange = axes.get_ylim()
#         plt.plot([pheno['TOM'],pheno['TOM']],[yrange[0],yfit[pheno['TOM']]], color='black', marker='', linestyle='--')
#         plt.plot([pheno['SOS'], pheno['SOS']], [yrange[0], yfit[pheno['SOS']]], color='black', marker='', linestyle='--')
#         plt.plot([pheno['SEN'], pheno['SEN']], [yrange[0], yfit[pheno['SEN']]], color='black', marker='', linestyle='--')
#         plt.plot([pheno['EOS'], pheno['EOS']], [yrange[0], yfit[pheno['EOS']]], color='black', marker='', linestyle='--')
#         axes.set_ylim(yrange)
#         plt.xlabel('days from ' + str(doy[iMin]))
#         plt.ylabel('NDVI')
#         plt.show()
#     #plt.close()
#     #now remap days from 0 to actual deks
#     for t in ('SOS','SEN','EOS','TOM'):
#         pheno[t] = dek[iMin] + pheno[t]
#
#     return pheno

#for debug
#nd = np.asarray([0.15904797408374433, 0.16994167391935014, 0.17979442026883283, 0.18939303258872453, 0.18941543371047279, 0.1800814718112369, 0.1641944541146892, 0.14661307900589882, 0.13775140702059763, 0.13337666569964218, 0.12974251523063535, 0.12858324630113147, 0.12807671888598784, 0.12787158882119723, 0.12822805821487282, 0.12826435064307126, 0.1311482255101054, 0.13509884924088578, 0.1377338506914225, 0.139736418141379, 0.14335629532927183, 0.14673102214486022, 0.15644542036381612])
#doy = np.asarray([1, 17, 33, 49, 65, 81, 97, 113, 129, 145, 161, 177, 193, 209, 225, 241, 257, 273, 289, 305, 321, 337, 353])
#print(pheno_lta_mono_wrapper(nd,doy,prctSOS=20,prctEOS=20))