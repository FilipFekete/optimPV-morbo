import os
import pandas as pd
import matplotlib.pyplot as plt
# Directory containing the files
directory = '/home/lecorre/Desktop/optimPV/Data/matdata/'

plt.figure(1)
plt.figure(2)
# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.startswith('nk_') and filename.endswith('.csv'):
        print(filename)
        # Read the CSV file
        csv_path = os.path.join(directory, filename)
        # check if first element first row is digit in the csv file
        # read text file
        with open(csv_path, 'r') as file:
            first_line = file.readline().strip()
            print(first_line)
        if first_line[0].isdigit():
            # read the csv file
            df = pd.read_csv(csv_path,  sep=',', names=['lambda', 'n', 'k'])
        else:
            df = pd.read_csv(csv_path,  sep=',', header='infer')
        print(df.head())
        # print(df.head())
        # Convert to string and append e-9
        # change the first column to string
        # df.iloc[:,0] = df.iloc[:,0].astype(str)
        names = list(df.columns)
        df[names[0]] = df[names[0]].astype(str) + 'e-9'
        # for i in range(len(df)):
        #     df.iloc[i,0] = str(df.iloc[i,0]) + 'e-9'
        # df.iloc[:,0] = df.iloc[:,0].to_string() + 'e-9'
        # df.iloc[:,0] = 
        # df.iloc[:,0] = pd.to_numeric(df.iloc[:,0]) 
        
        # Add the header
        df.columns = ['lambda', 'n', 'k']

        # Save the DataFrame to a txt file with white space separation
        txt_path = os.path.join(directory, filename.replace('.csv', '.txt'))
        df.to_csv(txt_path, sep=' ', index=False)

        # Plot the data
        plt.figure(1)
        plt.plot(df['lambda'], df['n'], label=filename.split('_')[1].split('.')[0])
        # plt.legend()
        plt.figure(2)
        plt.plot(df['lambda'], df['k'], label=filename.split('_')[1].split('.')[0])
        # plt.legend()
        # plt.show()


plt.figure(1)
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('n')
plt.title('Refractive index')

plt.figure(2)
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('k')
plt.title('Extinction coefficient')
plt.show()
