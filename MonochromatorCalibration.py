import numpy as np
import cantrips as can
import scipy.optimize as optimize
import matplotlib.pyplot as plt

def readInBlackCometMeasurement(black_comet_file, data_dir = '', n_ignore = 2, delimiter = ' '):
    """
    Read in a data file taken with the Black Comet spectograph.
    """
    read_in_data = can.readInColumnsToList(data_dir + black_comet_file, n_ignore = n_ignore, verbose = 0 )
    read_in_data = [[float(elem) for elem in col] for col in read_in_data]
    return read_in_data

def getFitFunct(fit_funct = 'gauss', fit_bounds = (-np.inf, np.inf)):
    """
    Get the fitting function and associated parameter bounds used
       to fit spectra components.
    """
    if fit_funct in ['gauss', 'GAUSS', 'Gauss']:
        fit_funct = lambda waves, A, mu, sig, shift: A * np.exp(-((np.array(waves) - mu) ** 2.0) / sig ** 2.0 ) + shift
        if fit_bounds == (-np.inf, np.inf):
            fit_bounds = ( [1.0, 0.0, 1.0, -40000.0], [np.inf, np.inf, 500.0, 40000.0] )
    return (fit_funct, fit_bounds)

def fitData(xs, ys, fit_funct = 'gauss', fit_guess = None, show_fit = 1, fit_bounds = (-np.inf, np.inf)  ):
    """
    Try to fit a single component of a Black Comet
       spectrum, generally a single line/spectrum
       peak.  The fitting algorithm requires a
       decent initial guess.
    """
    fit_funct, fit_bounds = getFitFunct(fit_funct = fit_funct, fit_bounds = fit_bounds)

    try:
        first_fit = optimize.curve_fit(lambda waves, A, mu, sig: fit_funct(waves, A, mu, sig, 0.0), xs, ys, p0 = fit_guess[0:3]  , bounds = (fit_bounds[0][0:3], fit_bounds[1][0:3]), method = 'trf' )
        second_fit = optimize.curve_fit(fit_funct, xs, ys, p0 = list(first_fit[0]) + [fit_guess[3]], bounds = fit_bounds, method = 'trf' )
    except RuntimeError:
        print ('Unable to fit MC observation.  Returning initial guess')
        second_fit = [ init_guess ]
    if second_fit[0][1] > fit_bounds[1][1] or second_fit[0][1] < fit_bounds[0][1]:
        print ('FOR: [xs,  ys] = ' + str([xs,  ys] ))
        print ('BOUNDS were: fit_bounds = '+ str(fit_bounds))
        print ('FIT was: bounds = '+ str(second_fit))
    #print ('second_fit[0] = ' + str(second_fit[0]))
    if show_fit:
        plt.plot(xs, ys, c = 'k')
        plt.plot(xs, fit_funct(xs, *fit_guess), c = 'r')
        plt.plot(xs, fit_funct(xs, *first_fit[0], 0.0) , c = 'orange')
        plt.plot(xs, fit_funct(xs, *second_fit[0]), c = 'g')
        plt.show( )
    return second_fit

def getRefLampWavelengths(ref_lamp_str):
    """
    Read in a fixed catalogue of reference wavelengths, all in nm
        for commonly used lab sources.
    """
    known_ref_lamps = ['hg2','kr1']
    HG2_ref_wavelengths = [253.652, 296.728, 302.150, 313.155, 334.148, 365.015, 404.656, 407.783, 435.833, 546.074, 576.960, 579.066, 696.543, 706.722 , 714.704 , 727.294 , 738.398, 750.387 , 763.511, 772.376 , 794.818, 800.616, 811.531, 826.452, 842.465, 852.144, 866.794 , 912.297, 922.450 , ]
    #Curate a few that are redundant/not helpful
    HG2_ref_wavelengths = [365.015, 404.656, 435.833, 546.074, 576.960, 696.543, 706.722 , 714.704 , 727.294 , 738.398, 750.387 , 763.511, 772.376 , 794.818, 800.616, 811.531, 826.452, 842.465, 852.144, 866.794 , 912.297, 922.450 , ]
    KR1_ref_wavelengths = [556.222, 557.029, 587.096, 760.155, 768.525, 769.454, 785.482, 791.343, 805.950, 810.436, 819.006, 826.324, 829.911, 877.675, 892.869]
    if ref_lamp_str.lower() == 'hg2':
        return HG2_ref_wavelengths
    elif ref_lamp_str.lower() == 'kr1':
        return KR1_ref_wavelengths
    else:
        print ('Reference lamp id ' + str(ref_lamp_str) + ' has no stored set of wavelengths.  Currently saved reference lamps are: ' + str(known_ref_lamps) + '.  Exiting...')
        sys.exit()

if __name__ == "__main__":
    """
    Determines the wavelength correction for the Stubbs group
        monochromator.  Requires data sets to have been
        gathered, as described on the Stubbs wiki.  The script
        is run simply as:
        $ python MonochromatorCalibration.py
    As a user, you normally just need to update the
        mono_wavelengths, MC_measurement_files, and date_str
        variables.  Though other parameters could be changed
        (whether you the user uses a KR1 or an HG2 Reference
        lamp, for example).

    Run from command line as:
    $ python MonochromatorCalibration.py
    """
    mono_date_str = '20220411'
    ref_lamp_date_str = '20220317'
    dir_base = '/Users/sashabrownsberger/Documents/Harvard/physics/stubbs/'
    measurements_dir = dir_base + 'MonochromatorCal/' + mono_date_str + '/'

    #MC_measurement_files = ['BC_mono_' + str(wave_len) + 'nm_exp10000ms.SSM' for wave_len in range(425, 426, 25)] + ['BC_mono_' + str(wave_len) + 'nm_exp1000ms.SSM' for wave_len in range(450, 476, 25)] + ['BC_mono_' + str(wave_len) + 'nm_exp200ms.SSM' for wave_len in range(500, 551, 25)] + ['BC_mono_' + str(wave_len) + 'nm_exp100ms.SSM' for wave_len in range(575, 626, 25)] + ['BC_mono_' + str(wave_len) + 'nm_exp50ms.SSM' for wave_len in range(650, 676, 25)] + ['BC_mono_' + str(wave_len) + 'nm_exp20ms.SSM' for wave_len in range(700, 726, 25)] + ['BC_mono_' + str(wave_len) + 'nm_exp10ms.SSM' for wave_len in range(750, 926, 25)] + ['BC_mono_' + str(wave_len) + 'nm_exp100ms.SSM' for wave_len in range(950, 1026, 25)]
    MC_measurement_files = ['mono_' + str(wave_len) + 'nm_10s.SSM' for wave_len in range(450, 450, 50)] + ['mono_' + str(wave_len) + 'nm_1s.SSM' for wave_len in range(450, 501, 50)] + ['mono_' + str(wave_len) + 'nm_100ms.SSM' for wave_len in range(550, 651, 50)] + ['mono_' + str(wave_len) + 'nm_10ms.SSM' for wave_len in range(700, 851, 50)] + ['mono_' + str(wave_len) + 'nm_100ms.SSM' for wave_len in range(900, 951, 50)] +  ['mono_' + str(wave_len) + 'nm_1000ms.SSM' for wave_len in range(1000, 1051, 50)]
    MC_measurements = [ readInBlackCometMeasurement(measurement_file, data_dir = measurements_dir)  for measurement_file in MC_measurement_files ]
    #mono_wavelengths = list(range(425, 1026, 25))
    mono_wavelengths = list(range(450, 1051, 50))


    wavelength_conv_fit_order = 2

    #Calibration lights include: [KR1, HG2, ]
    ref_lamp_str = 'KR1'
    ref_lamp_wavelengths = getRefLampWavelengths(ref_lamp_str)
    ref_lamp_dir = dir_base + 'MonochromatorCal/' + ref_lamp_date_str + '/'
    save_fig_name = 'monochromator_wavelength_determination_' + mono_date_str + '_1.pdf'

    #This is the wavelength around which the polynomial will be centered.
    #  The polynomial terms are most intuitive if fit_central_wavelength_nm
    #  is set to 0, but the fit can be better if the central value is in
    #  the middle of the fitted wavelength range.
    fit_central_wavelength_nm = 0
    wavelength_indeces = [ np.argmin(np.abs(np.array(MC_measurements[i][0]) - mono_wavelengths[i])) for i in range(len(MC_measurements)) ]
    MC_fit_pix_width = 100
    fitted_xs_set = [[] for i in range(len(MC_measurements))]
    fitted_ys_set = [[] for i in range(len(MC_measurements))]

    MC_fits = [0.0 for i in range(len(MC_measurements))]
    for i in range(len(MC_measurements)):
        wavelength = mono_wavelengths[i]
        MC_measurement = MC_measurements[i]
        wave_index = wavelength_indeces[i]
        n_points = len(MC_measurement[0])
        print ('Working on monochromator wavelength ' + str(wavelength))
        fit_funct_type = 'gauss'
        fit_xs = MC_measurement[0][max(wave_index - MC_fit_pix_width, 0):min(wave_index + MC_fit_pix_width, n_points)]
        fit_ys = MC_measurement[1][max(wave_index - MC_fit_pix_width, 0):min(wave_index + MC_fit_pix_width, n_points )]
        fitted_xs_set[i] = fit_xs
        fitted_ys_set[i] = fit_ys
        amp_guess = np.max(MC_measurement[1][max(wave_index - MC_fit_pix_width, 0):min(wave_index + MC_fit_pix_width, n_points )]) - np.median(MC_measurement[1][max(wave_index - MC_fit_pix_width, 0):min(wave_index + MC_fit_pix_width, n_points )])
        mu_guess = mono_wavelengths[i]
        sig_guess = 20.0
        floor_guess = np.median(MC_measurement[1][max(wave_index - MC_fit_pix_width, 0):min(wave_index + MC_fit_pix_width, n_points )])
        init_guess = [amp_guess, fit_xs[np.argmax(fit_ys)] , sig_guess, floor_guess ]
        MC_fit = fitData(fit_xs, fit_ys, fit_guess =  init_guess, show_fit = 0, fit_funct = fit_funct_type )[0]
        MC_fits[i] = MC_fit
        fit_funct, fit_bounds = getFitFunct(fit_funct = fit_funct_type )
        data_plot = plt.plot(fit_xs, fit_ys, c= 'k')[0]
        guess_fit_plot = plt.plot(fit_xs, fit_funct(fit_xs, *init_guess), c = 'r')[0]
        full_fit_plot = plt.plot(fit_xs, fit_funct(fit_xs, *MC_fit), c = 'g')[0]
        plt.title('MC Fitted curve')
        plt.xlabel('Black Comet Wavelength (nm)')
        plt.ylabel('Black Comet Intensity (ADU)')
        plt.legend([data_plot, guess_fit_plot, full_fit_plot], ['Monochromator Data', 'Initial Fit Guess', 'Actual Fit'])
        plt.draw()
        plt.pause(0.2)
        plt.cla()
    plt.close('all')
    #print ('[len(mono_wavelengths), len(MC_fits)] = ' + str([len(mono_wavelengths), len(MC_fits)] ))
    monochrom_black_comet_wavelengths = [MC_fit[1] for MC_fit in MC_fits]
    Mono_to_BC_fit_params = np.polyfit(np.array(mono_wavelengths) - fit_central_wavelength_nm, monochrom_black_comet_wavelengths,  wavelength_conv_fit_order )
    Mono_to_BC_funct = lambda wavelengths: np.poly1d(Mono_to_BC_fit_params)(np.array(wavelengths) - fit_central_wavelength_nm)
    BC_to_Mono_fit_params = np.polyfit(np.array(monochrom_black_comet_wavelengths) - fit_central_wavelength_nm, mono_wavelengths, wavelength_conv_fit_order )
    BC_to_Mono_funct = lambda wavelengths: np.poly1d(BC_to_Mono_fit_params)(np.array(wavelengths) - fit_central_wavelength_nm)
    #print ('monochrom_black_comet_wavelengths = ' + str(monochrom_black_comet_wavelengths ))


    ref_lamp_ref_strs = [ str(int(ref_wave)) + 'p' +  ('00' if int(round((ref_wave - int(ref_wave)) * 1000)) < 10  else '0' if int(round((ref_wave - int(ref_wave)) * 1000)) < 100 else '') + str(int(round((ref_wave - int(ref_wave)) * 1000))) for ref_wave in ref_lamp_wavelengths ]
    ref_lamp_data_files = ['BC_' + ref_lamp_str + '_exp1ms_'  + index + '.SSM' for index in ['A'] ]
    all_ref_lamp_data = [readInBlackCometMeasurement(ref_lamp_data_file, data_dir = ref_lamp_dir) for ref_lamp_data_file in ref_lamp_data_files]
    ref_lamp_data = [all_ref_lamp_data[0][0], np.mean([single_ref_lamp_data_set[1] for single_ref_lamp_data_set in all_ref_lamp_data], axis = 0)]
    wavelength_step = ref_lamp_data[0][1] - ref_lamp_data[0][0]
    fit_width_nm = 4
    fit_width_pix = int(fit_width_nm / wavelength_step)


    #Now we need to fit this to that
    closest_match_wavelength_indeces = [np.argmin(np.abs(np.array(ref_lamp_data[0]) - ref_lamp_wave)) for ref_lamp_wave in ref_lamp_wavelengths]
    #print ('closest_match_wavelength_indeces  = ' + str(closest_match_wavelength_indeces ))
    #print ('[[max(0, closest_match_index - fit_width_pix), min(closest_match_index + fit_width_pix, len(HG2_data[0]))] for closest_match_index in closest_match_wavelength_indeces ] = ' + str([[max(0, closest_match_index - fit_width_pix), min(closest_match_index + fit_width_pix, len(HG2_data[0]))] for closest_match_index in closest_match_wavelength_indeces ]))
    fit_data_segments = [[ ref_lamp_data[0][max(0, closest_match_index - fit_width_pix): min(closest_match_index + fit_width_pix, len(ref_lamp_data[0]))], ref_lamp_data[1][max(0, closest_match_index - fit_width_pix): min(closest_match_index + fit_width_pix, len(ref_lamp_data[1])) ] ]
                        for closest_match_index in closest_match_wavelength_indeces ]
    single_ref_lamp_line_fits = [fitData(fit_data_segments[i][0], fit_data_segments[i][1], fit_guess = [np.max(fit_data_segments[i][1]) - np.median(fit_data_segments[i][1]), ref_lamp_wavelengths[i], 3.0, np.median(ref_lamp_data[1])], show_fit = 0,  fit_bounds = ([0.0, 0.0, 1.0, 0.0], [66000, 2000.0, 4.0, 66000]), fit_funct = fit_funct_type)[0] for i in range(len(ref_lamp_wavelengths)) ]
    #print ('single_HG_line_fits = ' + str(single_HG_line_fits))
    ref_lamp_fitted_wavelengths = [single_ref_lamp_line_fit[1] for single_ref_lamp_line_fit in single_ref_lamp_line_fits  ]
    #print ('HG2_fitted_wavelengths = ' + str(HG2_fitted_wavelengths))
    #HG2_wavelength_diffs = [HG2_fitted_wavelengths[i] - HG2_ref_wavelengths[i] for i in range(len(HG2_ref_wavelengths))]
    ref_lamp_to_BC_fit_params = np.polyfit(np.array(ref_lamp_wavelengths) - fit_central_wavelength_nm, ref_lamp_fitted_wavelengths, wavelength_conv_fit_order )
    #print ('HG2_to_BC_fit_params = ' + str(HG2_to_BC_fit_params))
    true_wavelength_to_black_comet_funct = lambda wavelengths: np.poly1d(ref_lamp_to_BC_fit_params)(np.array(wavelengths) - fit_central_wavelength_nm)
    BC_to_ref_lamp_fit_params = np.polyfit(np.array(ref_lamp_fitted_wavelengths) - fit_central_wavelength_nm, ref_lamp_wavelengths,  wavelength_conv_fit_order )
    ref_lamp_to_BC_funct = lambda wavelengths: np.poly1d(ref_lamp_to_BC_fit_params)(np.array(wavelengths) - fit_central_wavelength_nm)
    BC_to_ref_lamp_funct = lambda wavelengths: np.poly1d(BC_to_ref_lamp_fit_params)(np.array(wavelengths) - fit_central_wavelength_nm)

    """
    f, axarr = plt.subplots(3,2, figsize = (8,10), sharex = 'col')
    axarr[0,0].scatter(mono_wavelengths, [MC_fit[1] for MC_fit in MC_fits], c = 'r')
    axarr[0,0].plot(mono_wavelengths, Mono_to_BC_funct(mono_wavelengths), c = 'k', linestyle = '--')
    axarr[1,0].scatter(mono_wavelengths, np.array([MC_fit[1] for MC_fit in MC_fits]) - np.array(mono_wavelengths), c = 'r')
    axarr[2,0].scatter(mono_wavelengths, np.array([MC_fit[1] for MC_fit in MC_fits]) - Mono_to_BC_funct(mono_wavelengths), c = 'r')
    axarr[2,0].plot(mono_wavelengths, [0.0 for wave in mono_wavelengths], c = 'k', linestyle = '--')
    axarr[2,0].set_xlabel('MC Wavelength (nm)')
    axarr[0,0].set_ylabel('BC Wavelength (nm)')
    axarr[1,0].set_ylabel('Measured (BC) - Purported (MC) (nm)')
    axarr[2,0].set_ylabel('Fit residual (nm)')
    axarr[0,1].scatter(ref_lamp_fitted_wavelengths, ref_lamp_wavelengths, c = 'r')
    axarr[0,1].plot(ref_lamp_fitted_wavelengths, BC_to_ref_lamp_funct(ref_lamp_fitted_wavelengths), c = 'k', linestyle = '--')
    axarr[1,1].scatter(ref_lamp_fitted_wavelengths, np.array(ref_lamp_wavelengths) - np.array(ref_lamp_fitted_wavelengths), c = 'r')
    axarr[2,1].scatter(ref_lamp_fitted_wavelengths, np.array(ref_lamp_wavelengths) - BC_to_ref_lamp_funct(ref_lamp_fitted_wavelengths), c = 'r')
    axarr[2,1].plot(ref_lamp_fitted_wavelengths, [0.0 for wave in ref_lamp_fitted_wavelengths], c = 'k', linestyle = '--')
    axarr[2,1].set_xlabel('BC Wavelength (nm)')
    axarr[0,1].set_ylabel('Ref lamp Wavelength (nm)')
    axarr[1,1].set_ylabel('True (Lamp) - Measured (BC) (nm)')
    axarr[2,1].set_ylabel('Fit residual (nm)')
    plt.tight_layout()
    plt.show()
    """

    ref_lamp_to_Mono_params = (ref_lamp_wavelengths, BC_to_ref_lamp_funct)
    True_to_Mono_funct = lambda wavelengths: BC_to_Mono_funct(ref_lamp_to_BC_funct(wavelengths))
    True_to_Mono_fit_params = np.polyfit(np.array(ref_lamp_fitted_wavelengths) - fit_central_wavelength_nm, [True_to_Mono_funct(ref_lamp_fitted_wavelength) for ref_lamp_fitted_wavelength in ref_lamp_fitted_wavelengths],  wavelength_conv_fit_order)
    Mono_to_True_funct = lambda wavelengths: BC_to_ref_lamp_funct(Mono_to_BC_funct(wavelengths))
    Mono_to_True_fit_params = np.polyfit(np.array(mono_wavelengths) - fit_central_wavelength_nm, [Mono_to_True_funct(wave) for wave in mono_wavelengths],  wavelength_conv_fit_order)
    Mono_to_True_polyfit = np.poly1d(Mono_to_True_fit_params)
    true_fit_funct = lambda waves: Mono_to_True_polyfit(np.array(waves) - fit_central_wavelength_nm )

    f, axarr = plt.subplots(4, 1, figsize = (20, 12))
    x_lims = [ mono_wavelengths[0] - (mono_wavelengths[-1] - mono_wavelengths[0]) * 0.05, mono_wavelengths[-1] + (mono_wavelengths[-1] - mono_wavelengths[0]) * 0.05 ]
    y_lims = [0.0, 70000]
    frequency_of_mono_fits_to_show = 1
    mono_plot = [axarr[0].plot(fitted_xs_set[i], fitted_ys_set[i], c = 'blue') for i in range(len(mono_wavelengths))][0][0]
    mono_ref = axarr[0].vlines([mono_wavelengths[i] for i in range(len(mono_wavelengths)) if i % frequency_of_mono_fits_to_show == 0], 0.0, 66000, linestyles = 'dashed', colors = 'gray', alpha = 0.5)
    [axarr[0].text(MC_fits[i][1], y_lims[1] * 0.8, str(mono_wavelengths[i]) + 'nm', horizontalalignment = 'center', verticalalignment = 'top', rotation = 90) for i in range(len(mono_wavelengths)) if i % frequency_of_mono_fits_to_show == 0]
    axarr[0].vlines([mono_wavelengths[i] for i in range(len(mono_wavelengths)) if i % frequency_of_mono_fits_to_show == 0], 0.0, 66000, linestyles = 'dashed', colors = 'gray', alpha = 0.5)
    axarr[0].set_ylabel('BC Integrated ADU')
    axarr[0].set_xlabel('BC Reported Wavelength (nm)')
    axarr[0].legend([mono_plot, mono_ref], ['Monochromator Spectra', 'Set Monochromator Waves'], ncol = 1)
    axarr[0].set_xlim(x_lims)
    axarr[0].set_ylim(y_lims)

    ref_lamp_plot = axarr[1].plot(ref_lamp_data[0], ref_lamp_data[1], c = 'orange')[0]
    ref_lamp_ref_lines = axarr[1].vlines(ref_lamp_wavelengths, 0.0, 5000, linestyles = 'dotted', colors = 'gray', alpha = 0.5, transform=axarr[2].get_xaxis_transform())
    axarr[1].set_yscale('log')
    axarr[1].set_xlabel('BC Reported Wavelength')
    axarr[1].set_ylabel('BC Integrated ADU')
    axarr[1].legend([ref_lamp_plot, ref_lamp_ref_lines], [ref_lamp_str + ' BC spectrum', 'Known ' + ref_lamp_str + ' lines'])
    axarr[1].set_xlim(x_lims)
    axarr[1].set_ylim(y_lims)

    mono_plot = [axarr[2].plot(BC_to_Mono_funct(fitted_xs_set[i]), fitted_ys_set[i], c = 'blue') for i in range(len(mono_wavelengths))][0][0]
    mono_ref = axarr[2].vlines([mono_wavelengths[i] for i in range(len(mono_wavelengths)) if i % frequency_of_mono_fits_to_show == 0], 0.5, 1, linestyles = 'dashed', colors = 'gray', alpha = 0.5, transform=axarr[2].get_xaxis_transform())
    [axarr[2].text(MC_fits[i][1], MC_fits[i][0] + MC_fits[i][3], str(mono_wavelengths[i]) + ' nm', horizontalalignment = 'right', rotation = 90) for i in range(len(mono_wavelengths)) if i % frequency_of_mono_fits_to_show == 0]
    ref_lamp_plot = axarr[2].plot(BC_to_Mono_funct(ref_lamp_data[0]), -np.array(ref_lamp_data[1]) , c = 'orange')[0]
    ref_lamp_ref_lines = axarr[2].vlines(True_to_Mono_funct(ref_lamp_wavelengths), 0, 0.5, linestyles = 'dotted', colors = 'gray', alpha = 0.5, transform=axarr[2].get_xaxis_transform())
    [axarr[2].text(can.round_to_n(True_to_Mono_funct(ref_lamp_wavelengths[i]), 3), -y_lims[1] * 0.8, str(can.round_to_n(True_to_Mono_funct(ref_lamp_wavelengths[i]), 4)) + ' nm', horizontalalignment = 'right', verticalalignment = 'bottom', rotation = 90) for i in range(len(ref_lamp_wavelengths)) ]
    axarr[2].set_ylabel('BC Integrated ADU')
    axarr[2].set_xlabel('Inferred Monochromator Wavelength (nm)')
    axarr[2].legend([mono_plot, mono_ref, ref_lamp_plot, ref_lamp_ref_lines], ['Monochromator Spectra', 'Set Monochromator Waves', ref_lamp_str + ' Spectra', ref_lamp_str + ' Strong Lines'], ncol = 1)
    axarr[2].set_xlim(x_lims)
    axarr[2].set_ylim([- y_lims[1], y_lims[1]])

    axarr[3].scatter(true_fit_funct(mono_wavelengths), [MC_fit[2] for MC_fit in MC_fits], marker = 'x', c = 'k')
    axarr[3].set_xlabel('Inferred Monochromator wavelength (nm)')
    axarr[3].set_ylabel(r'$\sigma$ of Gauss. fit (nm)')
    axarr[3].set_xlim(x_lims)

    """
    show_correction = 0
    if show_correction:
        #ref_corrected_monochrom_data_files = [ 'MC_spectrum_2000msInt_targetWave313p155.SSM', 'MC_spectrum_2000msInt_targetWave365p015.SSM', 'MC_spectrum_2000msInt_targetWave404p656.SSM', 'MC_spectrum_2000msInt_targetWave435p833.SSM', 'MC_spectrum_2000msInt_targetWave546p074.SSM', 'MC_spectrum_2000msInt_targetWave579p06.SSM', 'MC_spectrum_2000msInt_targetWave696p543.SSM', 'MC_spectrum_2000msInt_targetWave727p294.SSM', 'MC_spectrum_2000msInt_targetWave750p387.SSM',
        #                                      'MC_spectrum_2000msInt_targetWave763p511.SSM', 'MC_spectrum_2000msInt_targetWave772p376.SSM', 'MC_spectrum_2000msInt_targetWave800p616.SSM', 'MC_spectrum_2000msInt_targetWave811p531.SSM', 'MC_spectrum_2000msInt_targetWave826p452.SSM', 'MC_spectrum_2000msInt_targetWave842p465.SSM', 'MC_spectrum_2000msInt_targetWave912p297.SSM']
        ref_corrected_monochrom_data_files = ['Mono_' + ref_lamp_ref_str + '_TargetWave_500ms.SSM' for ref_lamp_ref_str in ref_lamp_ref_strs[0:3]] + ['Mono_' + ref_lamp_ref_str + '_TargetWave_100ms.SSM' for ref_lamp_ref_str in ref_lamp_ref_strs[3:]]
        ref_corrected_monochrom_data_sets = [readInBlackCometMeasurement(ref_data_file, data_dir = ref_lamp_dir)  for ref_data_file in ref_corrected_monochrom_data_files]
        #ref_uncorrected_monochrom_data_files = [ 'MC_spectrum_2000msInt_goWave313p155.SSM', 'MC_spectrum_2000msInt_goWave365p015.SSM', 'MC_spectrum_2000msInt_goWave404p656.SSM', 'MC_spectrum_2000msInt_goWave435p833.SSM', 'MC_spectrum_2000msInt_goWave546p074.SSM', 'MC_spectrum_2000msInt_goWave579p06.SSM', 'MC_spectrum_2000msInt_goWave696p543.SSM', 'MC_spectrum_2000msInt_goWave727p294.SSM', 'MC_spectrum_2000msInt_goWave750p387.SSM',
        #                                      'MC_spectrum_2000msInt_goWave763p511.SSM', 'MC_spectrum_2000msInt_goWave772p376.SSM', 'MC_spectrum_2000msInt_goWave800p616.SSM', 'MC_spectrum_2000msInt_goWave811p531.SSM', 'MC_spectrum_2000msInt_goWave826p452.SSM', 'MC_spectrum_2000msInt_goWave842p465.SSM', 'MC_spectrum_2000msInt_goWave912p297.SSM']
        ref_uncorrected_monochrom_data_files = ['Mono_' + ref_lamp_ref_str + '_GoWave_500ms.SSM' for ref_lamp_ref_str in ref_lamp_ref_strs[0:3]] + ['Mono_' + ref_lamp_ref_str + '_GoWave_100ms.SSM' for ref_lamp_ref_str in ref_lamp_ref_strs[3:]]
        ref_uncorrected_monochrom_data_sets = [readInBlackCometMeasurement(ref_data_file, data_dir = ref_lamp_dir)  for ref_data_file in ref_uncorrected_monochrom_data_files]
        ref_corrected_max_indeces = [np.argmax(ref_data_set[1]) for ref_data_set in ref_corrected_monochrom_data_sets]
        ref_uncorrected_max_indeces = [np.argmax(ref_data_set[1]) for ref_data_set in ref_uncorrected_monochrom_data_sets]
        print ('ref_corrected_max_indeces = ' + str(ref_corrected_max_indeces))
        print ('[len(ref_corrected_monochrom_data_sets), len(ref_corrected_max_indeces)] = ' + str([len(ref_corrected_monochrom_data_sets), len(ref_corrected_max_indeces)]))
        correction_sample_distance = 10
        ref_corrected_normalizations = [np.max(ref_lamp_data[1][max(ref_corrected_max_indeces[i]-correction_sample_distance, 0):min(ref_corrected_max_indeces[i]+correction_sample_distance, len(ref_lamp_data[1]))]) / ref_corrected_monochrom_data_sets[i][1][ref_corrected_max_indeces[i]]  for i in range(len(ref_corrected_monochrom_data_sets))]
        ref_uncorrected_normalizations = [np.max(ref_lamp_data[1][max(ref_uncorrected_max_indeces[i]-correction_sample_distance, 0):min(ref_uncorrected_max_indeces[i]+correction_sample_distance, len(ref_lamp_data[1]))]) / ref_uncorrected_monochrom_data_sets[i][1][ref_uncorrected_max_indeces[i]]  for i in range(len(ref_uncorrected_monochrom_data_sets))]
        print ('ref_corrected_normalizations = ' + str(ref_corrected_normalizations ))
        ref_corrected_lines = [axarr[1].plot(ref_corrected_monochrom_data_sets[i][0][max(ref_corrected_max_indeces[i]-correction_sample_distance, 0):min(ref_corrected_max_indeces[i]+correction_sample_distance, len(ref_lamp_data[1]))], np.array(ref_corrected_monochrom_data_sets[i][1])[max(ref_corrected_max_indeces[i]-correction_sample_distance, 0):min(ref_corrected_max_indeces[i]+correction_sample_distance, len(ref_lamp_data[1]))] * ref_corrected_normalizations[i], c = 'green', alpha = 0.5)[0] for i in range(len(ref_corrected_monochrom_data_sets))][0]
        ref_uncorrected_lines = [axarr[1].plot(ref_uncorrected_monochrom_data_sets[i][0][max(ref_uncorrected_max_indeces[i]-correction_sample_distance, 0):min(ref_uncorrected_max_indeces[i]+correction_sample_distance, len(ref_lamp_data[1]))], np.array(ref_uncorrected_monochrom_data_sets[i][1])[max(ref_uncorrected_max_indeces[i]-correction_sample_distance, 0):min(ref_uncorrected_max_indeces[i]+correction_sample_distance, len(ref_lamp_data[1]))] * ref_uncorrected_normalizations[i], c = 'red', alpha = 0.5)[0] for i in range(len(ref_uncorrected_monochrom_data_sets))][0]
        [axarr[1].text(single_ref_lamp_line_fits[i][1], 60000, str(can.round_to_n(ref_lamp_wavelengths[i], 3)) + 'nm', horizontalalignment = 'center', verticalalignment = 'top', rotation = 90) for i in range(len(single_ref_lamp_line_fits)) ]
        axarr[1].set_ylabel('BC Integrated ADU')
        axarr[1].set_xlabel('BC Reported Wavelength (nm)')
        axarr[1].legend([ref_lamp_plot, ref_lamp_ref_lines, ref_corrected_lines, ref_uncorrected_lines], ['HG-2 Spectra', 'HG-2 Strong Lines', 'Monochromator, corrected', 'Monochromator, uncorrected'], ncol = 1)
    """

    plt.tight_layout()
    plt.savefig(measurements_dir + save_fig_name)
    #plt.show()

    #print ('True wavelength (W_T) to monochromator wavelength (W_L) fit funct (both in nm): W_M = (' + str(True_to_Mono_fit_params[0]) + ')(W_T - ' + str(fit_central_wavelength_nm) + ')^2 + (' + str(True_to_Mono_fit_params[1]) + ')(W_T - ' + str(fit_central_wavelength_nm) + ') + ' + str(True_to_Mono_fit_params[2]) )
    #print ('True_to_Mono_fit_params = ' + str(True_to_Mono_fit_params ))
    print ('Monochromator wavelength (W_M) to true wavelength (W_T) fit funct (both in nm): W_T = (' + str(can.round_to_n(Mono_to_True_fit_params[0], 5)) + ')(W_M - ' + str(fit_central_wavelength_nm) + ')^2 + (' + str(can.round_to_n(Mono_to_True_fit_params[1], 5)) + ')(W_M - ' + str(fit_central_wavelength_nm) + ') + ' + str(can.round_to_n(Mono_to_True_fit_params[2], 5)) )
    print ('By that scale, the fit wavelengths for the reference lamp are: ')
    for ref_lamp_wave in ref_lamp_wavelengths:
        print ('Mono claimed: ' + str(ref_lamp_wave) + ' nm  =====> is actually: ' + str(can.round_to_n(true_fit_funct(ref_lamp_wave), 6)) + ' nm')
