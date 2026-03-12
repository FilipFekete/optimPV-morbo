'''Update the version number in the parameter files for SIMsalabim.'''
import os
import subprocess

# Define the path to SIMsalabim parameter files
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
Data_dir = os.path.join(main_dir, 'Data')
SIMsalabim_dir = os.path.join(main_dir, 'SIMsalabim')

# get the Updater tool from SIMsalabim and compile it with fpc
Updater_path = os.path.join(SIMsalabim_dir, 'Tools', 'Updater')
is_windows = (os.name == 'nt')
subprocess.check_call(['fpc','Updater.pas'], encoding='utf8', cwd=Updater_path, shell=is_windows)


Dir_to_update = [os.path.join(Data_dir,'OPV_degradation'),os.path.join(Data_dir,'pero_JV_QFLS'),os.path.join(Data_dir,'simsalabim_test_inputs','fakeOPV'),os.path.join(Data_dir,'simsalabim_test_inputs','fakePerovskite'),os.path.join(Data_dir,'simsalabim_test_inputs','JVrealOPV'),os.path.join(Data_dir,'simsalabim_test_inputs','JVrealPerovskite'),os.path.join(Data_dir,'simsalabim_test_inputs','JVsimpleLayerPV')]


for dir_ in Dir_to_update:
    # check for a file that starts with 'simulation_setup' in the directory to update
    simulation_setup_file_list = [file for file in os.listdir(dir_) if file.startswith('simulation_setup')]
    if len(simulation_setup_file_list) == 0:
        print(f'No simulation_setup file found in {dir_}. Skipping this directory.')
        continue
    
    for simulation_setup_file in simulation_setup_file_list:
        if is_windows:
            #copy the compiled Updater.exe to the directory to update
            subprocess.check_call(['copy', os.path.join(Updater_path, 'Updater.exe'), dir_], shell=True)
            #run the Updater.exe in the directory to update
            subprocess.check_call(['Updater.exe', simulation_setup_file], shell=True, cwd=dir_)
            # delete the Updater.exe after running
            subprocess.check_call(['del', os.path.join(dir_, 'Updater.exe')], shell=True)
        else:
            #copy the compiled Updater to the directory to update
            subprocess.check_call(['cp', os.path.join(Updater_path, 'Updater'), dir_])
            #run the Updater in the directory to update
            res = subprocess.run(["./Updater", simulation_setup_file], cwd=dir_, check=False)
            # delete the Updater after running
            subprocess.check_call(['rm', os.path.join(dir_, 'Updater')])