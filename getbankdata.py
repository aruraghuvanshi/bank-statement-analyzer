import pandas as pd
import cv2
from PIL import Image
import pytesseract as pt
import glob
from pdf2image import convert_from_path
import os
import shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from time import time
import re
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract as pt
import tabula
import warnings
warnings.filterwarnings('ignore')


class GetBankData:

    def __init__(self, pdf_file):

        pt.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
        self.axis_flag = False
        self.hdfc_flag = False
        self.kotak_flag = False
        self.icici_flag = False
        self.pdf_file = pdf_file
        self.output_csv_name = 'Output\\extraction.csv'
        self.format1 = True
        self.format2 = False
        self.bank_name = ''
        self.text = ''
        self.mapping = {
            'BRN': 'CC',
            'UPI': 'UPI',
            'ECOM': 'ONLINE',
            'POS': 'SWIPE',
            'NEFT': 'NEFT',
            'RTGS': 'RTGS',
            'VISA': 'SWIPE',
            'INB': 'NETBANK',
            'IMPS': 'IMPS',
            'BY': 'CASHDEP',
            'PUR': 'CARDCVV',
            'Int': 'INTEREST',
            'Ban': 'GPAY',
            'GST': 'TAX',
            'MOB': 'NETBANK',
            'Consolidated': 'TAX',
            'Bank': 'GPAY',
            'ATM': 'CASHWD',
            'Dr': 'TAX',
            'Mah': 'GPAY',
            'CTF': 'CARDCVV',
            'TRF': 'TRF',
            'Ba': 'UPI',
            'PCD': 'CARDCVV',
            'Visa': 'CREDIT',
            'Bank Rewarde': 'CREDIT',
            'Salary': 'CREDIT'}

    def clear_directory(self, allfiles=True):

        '''Removes files from previous run from input directory
        before each run
        '''
        print('\n> \033[0;35mInput Directory cleared of items.\033[0m\n')

        if allfiles:
            path = 'Input/ImageFiles/*.*'
        else:
            path = 'Input/ImageFiles/*.png'

        files = glob.glob(path)
        for f in files:
            os.remove(f)

    def convert_pdf_image(self):
        '''
        Returns objects at memory location. Can be accessed by indexing return.
        ex. >>> img = convert_pdf_image(pdf_file)

        '''
        print(f'> Converting {self.pdf_file} to PNG.')
        images = convert_from_path(self.pdf_file, poppler_path=r'E:\poppler-0.68.0\bin')
        #         for i in range(len(images)):
        #             images[i].save(f'Input/ImageFiles/page{i:02}.png', 'PNG')
        images[0].save(f'Input/ImageFiles/page00.png', 'PNG')
        #         print(f'> Number of generated Image files: {len(images)}\n')
        return images




    def get_bank_name(self, filename='Input/ImageFiles/page00.png'):
        '''
        Get the bank name from the first 20 characters of the first
        page of the converted pdf into png and raise bank flag'''
        print('\n> Searching for Bank Name...')
        self.text = pt.image_to_string(Image.open(filename))
        self.text = re.sub('\W+', ' ', self.text)
        self.text = self.text[:20].lower()
        self.bank_name = ' '
        if 'axis' in self.text:
            axis_flag = True
            self.bank_name = 'Axis Bank'
            print(f'> Bank Name: \033[0;32m{self.bank_name}\033[0m')
        elif 'hdfc' in self.text:
            hdfc_flag = True
            self.bank_name = 'HDFC Bank'
            print(f'> Bank Name: \033[0;32m{self.bank_name}\033[0m')
        elif 'kotak' in self.text:
            kotak_flag = True
            self.bank_name = 'Kotak Mahindra Bank'
            print(f'> Bank Name: \033[0;32m{self.bank_name}\033[0m')
        elif 'icici' in self.text:
            icici_flag = True
            self.bank_name = 'ICICI Bank'
        else:
            print('\033[0;31mNot found. Attempting to detect bank name through statement format.\033[0m')

        return self.text, self.bank_name




    def analyze_statement_format(self):

        dfs = tabula.read_pdf(self.pdf_file, pages='all', guess=False, stream=True, silent=True)
        tabula.convert_into(self.pdf_file, self.output_csv_name, output_format='csv', pages='all')
        #         self.dx = tabula.read_pdf(self.output_csv_name, pages='all', stream=True, guess=False)
        try:
            self.dx = pd.read_csv(self.output_csv_name)
            print(f"> \033[0;34mReading Extraction\033[0m (Try)...")
        except Exception as e:
            col_names = ["DATE", "value", "PARTICULARS", "DEBIT/CREDIT", "BALANCE"]
            self.dx = pd.read_csv(self.output_csv_name, names=col_names)
            self.dx = self.dx[2:]
            print(f'> \033[0;31mIn EXCEPT of analyze_statement_format - {e}\033[0m')

        print(f'\nStatement Information: \nColumns: {self.dx.columns}\nnum_columns: {len(self.dx.columns)}\n')

        if 'Value Date' in self.dx.columns or 'DR/CR' in self.dx.columns:
            self.format2 = True
            self.format1 = False
            self.bank_name = 'Axis Bank'
            print(f'\nStatement Format: \033[0;34m{self.bank_name} Type-II\033[0m')
            self.axis_flag = True
            print('in 1')

        elif 'Statement of Banking Account' in self.dx.columns:
            self.format2 = True
            self.format1 = False
            self.bank_name = 'Kotak Mahindra Bank'
            print(f'\nStatement Format: \033[0;34m{self.bank_name} Type-II\033[0m')
            print('in 2')

        elif "DEBIT/CREDIT(₹)" in self.dx.columns:
            self.format1 = True
            self.format2 = False
            self.bank_name = 'Kotak Mahindra Bank'
            print(f'\nStatement Format: \033[0;34m{self.bank_name} Type-I\033[0m')
            print('in 3')

        elif "Amt." in self.dx.columns or "Value Dt" in self.dx.columns:
            self.format1 = True
            self.format2 = False
            self.bank_name = 'HDFC Bank'
            print(f'\nStatement Format: \033[0;34m{self.bank_name} Type-I\033[0m')
            print('in 4')

        else:
            self.format1 = True
            self.format2 = False
            print(f'\nStatement Format: \033[0;34m{self.bank_name} Type-I\033[0m')
            print('in else 5')

        return self.dx




    def preprocess_ingest_axis_pdf(self):

        def create_credit_column(row):
            if row["DR/CR"] == 'CR':
                val = row["Amount(INR)"]
            else:
                val = 0
            return val

        def create_debit_column(row):
            if row["DR/CR"] == 'DR':
                val = row["Amount(INR)"]
            else:
                val = 0
            return val

        df = self.dx

        if self.format1:
            df.drop(['Chq No', 'Balance', 'Init.'], inplace=True, axis=1)
            df = df.dropna(how='all').reset_index(drop=True)
            df.drop(0, inplace=True)
            df.Debit.fillna(0, inplace=True)
            df.Credit.fillna(0, inplace=True)
            print(f'\nPdf to Csv dataset shape: {df.shape}')
            print(f'Shape before dropna: {df.shape}')
            df.dropna(inplace=True)
            print(f'Shape after dropna: {df.shape}')
            print(f'\nNA table: \n{df.isna().sum()}')

            df.rename(columns={'Tran Date': 'DATE', 'Debit': 'DR', 'Credit': 'CR',
                               'Particulars': 'PARTICULARS'}, inplace=True)

            print('\nOutputcsv created.')
            df.to_csv(f'Output\\axis_final_type1.csv', index=False)
            self.dx = df
            return df

        else:
            print(f'> Converting statement format to Type-I')
            df = df.drop(['Value Date', 'Branch Name', 'Chq No', "Balance(INR)"], axis=1)
            df = df.rename(columns={'Transaction Particulars': 'Particulars'})
            df = df.loc[1:]

            print(f'\nPdf to Csv dataset shape: {df.shape}')
            print(f'Shape before dropna: {df.shape}')
            df.dropna(inplace=True)
            print(f'Shape after dropna: {df.shape}')
            print(f'\nNA table: \n{df.isna().sum()}')

            df['Credit'] = df.apply(create_credit_column, axis=1)
            df['Debit'] = df.apply(create_debit_column, axis=1)
            df = df.drop(['DR/CR', "Amount(INR)"], axis=1)

            df.rename(columns={'Tran Date': 'DATE', 'Debit': 'DR', 'Credit': 'CR',
                               'Particulars': 'PARTICULARS'}, inplace=True)

            print('\n> Conversion complete. Outputcsv created.')
            df.to_csv(f'Output\\axis_final_type2.csv', index=False)
            self.dx = df
            return df




    def preprocess_ingest_kotak_pdf(self):

        def create_credit_column(row):
            if row["DR/CR"] == 'CR':
                val = row["Amount"]
            else:
                val = 0
            return val

        def create_debit_column(row):
            if row["DR/CR"] == 'DR':
                val = row["Amount"]
            else:
                val = 0
            return val

        df = self.dx
        t1 = time()
        if self.format2:

            print(f'> Parsing Kotak \033[0;34mType-II\033[0m statement...')
            dfs = tabula.read_pdf(self.pdf_file, pages='all', guess=False, stream=True, silent=True)
            tabula.convert_into(self.pdf_file, self.output_csv_name, output_format='csv', pages='all')

            df = pd.read_csv(self.output_csv_name)
            df = df.iloc[1:-2]
            df.rename(columns={'Statement of Banking Account': 'DATE',
                               'Unnamed: 1': 'PARTICULARS', 'Unnamed: 3': 'DR',
                               'Unnamed: 4': 'CR'}, inplace=True)
            df = df.drop(['Unnamed: 2', 'Unnamed: 5'], axis=1)
            df.DR.fillna(0, inplace=True)
            df.CR.fillna(0, inplace=True)
            print(f'Pdf to Csv dataset shape: {df.shape}')

            print(f'Shape before dropna: {df.shape}')
            print(df.isna().sum())
            df.dropna(inplace=True)
            print()
            print(df.isna().sum())
            print(f'Shape after dropna: {df.shape}')

            df.to_csv(f'Output\\kotak_final_type2.csv', index=False)
            print(f'\033[0;32mOutput file created in {round(time() - t1, 1)}s.\033[0m')
            self.dx = df
            return df

        else:
            try:
                print(f'> Parsing Kotak \033[0;34mType-I\033[0m statement...')
                dfs = tabula.read_pdf(self.pdf_file, pages='all', stream=True, guess=False, silent=True)
                tabula.convert_into(self.pdf_file, self.output_csv_name, output_format='csv', pages='all', stream=True)
                col_names = ["DATE", "value", "PARTICULARS", "DEBIT/CREDIT", "BALANCE"]
                df = pd.read_csv(self.output_csv_name, names=col_names)
                try:
                    df = df.loc[1:]
                except Exception as e:
                    print(f'\033[0;31mIn Except preprocess_ingest_kotak format-1 else block\033[0m - {e}')
                    df = df.iloc[1:]

                df = df.drop(['value'], axis=1)
                df = df.dropna().reset_index(drop=True)
                df['DR/CR'] = df["DEBIT/CREDIT"].apply(lambda x: 'CR' if '+' in x else 'DR')
                df['Amount'] = df['DEBIT/CREDIT'].apply(lambda x: (re.split('[- +]', x)[-1]))
                df['Amount'] = df['Amount'].apply(lambda o: o.split('.')[0])
                df['Amount'] = df.Amount.apply(lambda v: v.replace(',', '')).astype(int)
                df['CR'] = df.apply(create_credit_column, axis=1)
                df['DR'] = df.apply(create_debit_column, axis=1)
                df = df.drop(['DEBIT/CREDIT', 'BALANCE', 'Amount', 'DR/CR'], axis=1)
                print(f'> Conversion complete.')
                df.to_csv(f'Output\\kotak_final_type1.csv', index=False)
                print(f'\033[0;32mOutput file created in {round(time() - t1, 1)}s.\033[0m')
                self.dx = df
                return df
            except Exception as e:
                print(f'\033[0;31mEXCEPTION in preprocess_ingest_kotak\033[0m: - {e}')



    def preprocess_ingest_hdfc_pdf(self):

        t1 = time()
        if self.format1:
            try:
                print(f'> Parsing HDFC \033[0;34mType-I\033[0m statement...')
                dfs = tabula.read_pdf(self.pdf_file, pages='all', stream=True, guess=False, silent=True)
                tabula.convert_into(self.pdf_file, self.output_csv_name, output_format='csv', pages='all', stream=True)
                col_names = ["DATE", "PARTICULARS", "CHQ", "valuedt", "DR", "CR", "BALANCE"]
                df = pd.read_csv(self.output_csv_name, names=col_names)
                df = df.iloc[1:]
                odf = df.copy()
                df = df.drop(['CHQ', 'valuedt', 'BALANCE'], axis=1)
                df = df[df['DATE'].notna()].reset_index(drop=True)

                df.DR.fillna(0, inplace=True)
                df.CR.fillna(0, inplace=True)
                self.dx = df
                print(self.dx.columns)
                print(f'> Conversion complete.')
                df.to_csv(f'Output\\hdfc_final_type1.csv', index=False)
                print(f'\033[0;32mOutput file created in {round(time() - t1, 1)}s.\033[0m')
                return df
            except Exception as e:
                print(f'\033[0;31mEXCEPTION in preprocess_ingest_hdfc\033[0m: - {e}')
        else:
            print('> Parsing Type-II of HDFC Statement pdf')
            pass



    def preprocess_overall_bankdata(self):

        df = self.dx

        def remove_digits(org_string):
            pattern = r'[0-9]'
            mod_string = re.sub(pattern, '', str(org_string))
            return mod_string

        df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%d-%m-%Y')
        df.PARTICULARS = df.PARTICULARS.apply(remove_digits)
        df.PARTICULARS = df.PARTICULARS.str.split('/')
        df.PARTICULARS = df.PARTICULARS.apply(', '.join)
        df.PARTICULARS = df.PARTICULARS.str.replace(',', ' ')
        df.PARTICULARS = df.PARTICULARS.str.replace('-', ' ')
        df.PARTICULARS = df.PARTICULARS.str.replace(':', ' ')
        df.PARTICULARS = df.PARTICULARS.str.replace('.', ' ')
        df.PARTICULARS = df.PARTICULARS.str.replace('_', ' ')
        df.PARTICULARS = df.PARTICULARS.str.replace('\n', ' ', regex=True)
        df.PARTICULARS = df.PARTICULARS.str.replace('/', ' ', regex=True)
        df.PARTICULARS = df.PARTICULARS.str.replace('\r', ' ', regex=True)
        df.PARTICULARS = df.PARTICULARS.str.replace('+', ' ')
        df.PARTICULARS = df.PARTICULARS.str.replace('@', ' ')
        df.PARTICULARS = df.PARTICULARS.str.replace('*', '')
        df.PARTICULARS = df.PARTICULARS.apply(lambda x: x.strip())
        df.DR = df.DR.astype(str)
        df.CR = df.CR.astype(str)
        df['DR'] = df.DR.apply(lambda v: v.replace(',', '')).astype(float)
        df['CR'] = df.CR.apply(lambda v: v.replace(',', '')).astype(float)

        df['TYPE'] = df.PARTICULARS.apply(lambda x: x.split(' ')[0])
        df.TYPE = df.TYPE.str.replace('Int.Pd    to     ', 'INTP', regex=True)
        df = df.replace({'TYPE': self.mapping})
        df.loc[~df['TYPE'].isin(self.mapping.values()), 'TYPE'] = 'OTH'
        df.loc[df['PARTICULARS'].str.contains('Paytm'), 'TYPE'] = 'PAYTM'
        df.loc[df['PARTICULARS'].str.contains('GOOGLEPAY'), 'TYPE'] = 'GPAY'
        df.loc[df['PARTICULARS'].str.contains('MB'), 'TYPE'] = 'NETBANK'
        df.to_csv(f'Output\\master_output.csv', index=False)
        self.dx = df
        return df
