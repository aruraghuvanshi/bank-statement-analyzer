import random
from remi import App, start
import os
import remi.gui as tk
from creators import C
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from getbankdata import GetBankData
import glob
import pandas as pd
from user import User
from threading import Thread
from run_saved_model import load_test_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

pd.options.display.max_rows = 500
pd.options.display.max_columns = 40


'''
MAIN_V4.PY
Main_v2 was last stable build.
Main_v3 was copy of 2, but was not being tracked on git, 4 is created.
- Swap DR with CR in Kotak
- Check Dropdown of Banklist feasibility and add Kotak
- Modify code to add new frame for filtering options 
- Use the choosefile to choose file instead of hardcoded path
- Add HDFC to the bank preprocessing
- Clean out Directory
- Retrain Model
- Add Filters to view graphs on transaction TYPES and PRED_CAT
+ Do the hashlib for password or pickling
+ Revert to not having the entire code in main but modularize it.

'''


class BankStatementAnalyzer(App):

    def __init__(self, *args):

        super(BankStatementAnalyzer, self).__init__(*args, static_file_path={'path': './resx/'})

        self.bank_list = ['Select Bank', 'Axis Bank', 'HDFC Bank', 'Kotak Mahindra Bank', 'ICICI Bank']
        self.date = datetime.date.today().strftime('%d-%m-%Y')
        self.time = datetime.datetime.now().time()
        self.model_name = 'model_ann_98.h5'
        self.cv_name = 'vectorizer.sav'
        self.le_name = 'target_label_encoder.sav'


    def idle(self):
        pass


    def main(self):

        self.date = datetime.date.today().strftime('%d-%m-%Y')
        self.time = datetime.datetime.now().time()

        self.listview = tk.ListView()
        self.frame_left = tk.Container()
        self.frame_filter = tk.Container()
        self.frame_right = tk.Container()
        self.frame_header = tk.Container()
        self.frame_right_2 = tk.Container()
        self.master_user = pd.DataFrame()
        self.window = tk.Container()

        self.window.css_background_color = "azure"
        self.window.css_height = "100%"
        self.window.css_width = "100%"
        self.window.css_left = "0.0px"
        self.window.css_top = "0.0px"
        self.window.css_position = "absolute"

        self.frame_header_color = 'cornflowerblue'
        self.frame_left_color = 'ivory'
        self.frame_filter_color = 'whitesmoke'
        self.frame_footer_left_color = 'honeydew'
        self.frame_right_color = 'whitesmoke'
        self.frame_right_2_color = 'seashell'
        self.frame_login_register_color = 'azure'

        self.selected_bank = []
        self.registry_info = {}
        self.login_info = {}

        self.dt = pd.DataFrame()
        self.dx = pd.DataFrame()
        self.dr = pd.DataFrame()

        self.frame_header = C.create_container(self.window, 10, 90, 0, 0)
        self.frame_header.css_background_color = self.frame_header_color
        self.frame_header.css_top = "0%"

        self.frame_footer_left = C.create_container(self.window, 12, 20, 0, 87)
        self.frame_footer_left.css_background_color = self.frame_footer_left_color

        self.frame_left = C.create_container(self.window, 25, 20, 0, 10)
        self.frame_left.css_background_color = self.frame_left_color

        self.frame_filter = C.create_container(self.window, 50, 20, 0, 35)
        self.frame_filter.css_background_color = self.frame_filter_color

        self.frame_right = C.create_container(self.window, 75, 35, 21, 10)
        self.frame_right.css_background_color = self.frame_right_color

        self.frame_right_2 = C.create_container(self.window, 75, 33, 57, 10)
        self.frame_right_2.css_background_color = self.frame_right_2_color

        self.frame_login_register = C.create_container(self.window, 30, 10, 90, 0)
        self.frame_login_register.css_background_color = self.frame_login_register_color

        self.progress = C.create_progress(self.frame_footer_left, 5, 100, 0, 95, a=0, b=100)

        self.img1 = tk.Image()
        self.img = tk.Image('', height=300, margin='1px')
        self.upl =''


        lbl_header = C.create_label(self.frame_header, 20, 25, 10, 30, text='BANK STATEMENT ANALYZER',
                                    bg='cornflowerblue', fg='white')
        lbl_header.css_font_size = '18px'

        lbl_subheader = C.create_label(self.frame_header, 10, 20, 13.35, 60,
                                       text='-- Aru Raghuvanshi build 07042021',
                                       bg='cornflowerblue', fg='white')
        lbl_subheader.css_font_size = '12px'

        lbl_datetime = C.create_label(self.frame_header, 20, 7, 93, 40, text=f'Date: {self.date}',
                                      bg='cornflowerblue', fg='white', align='right')
        lbl_datetime.css_font_size = '14px'

        self.notif_1 = C.create_label(self.frame_footer_left, 6, 100, 0, 10, text='')
        self.notif_2 = C.create_label(self.frame_footer_left, 6, 100, 0, 60, text='')

        self.login_btn = C.create_button(self.window, 3, 7, 92, 1, text='Login',
                                         command=lambda x: self.login_clicked())
        self.register_btn = C.create_button(self.window, 3, 7, 92, 6, text='Register',
                                            command=lambda x: self.register_clicked())

        return self.window



    def clear_directory(self, emitter=None):

        '''Removes files from previous run from input directory
        before each run.
        emitter = None by default
        emitter = True  for /resx
        emitter = False for /Input PDF
        '''

        if emitter:
            path = 'resx/*.*'
        else:
            path = 'Input PDF/*.*'

        files = glob.glob(path)
        for f in files:
            os.remove(f)
        print(f'\nPath erased: {path}')
        print(f"\n> \033[0;35m'{path} cleared.\033[0m\n")



    def login_clicked(self):

        self.frame_left.empty()
        print(f'Login Button pressed')
        self.frame_login_register.empty()
        self.lbl_username = C.create_label(self.frame_login_register, 7, 40, 5, 40, text='Username:', bg='azure')
        self.lbl_pw = C.create_label(self.frame_login_register,  7, 40, 5, 50, text='Password:', bg='azure')
        self.username = C.create_entry(self.frame_login_register, 7, 52, 40, 40, fg='black', input_type='regular',
                                       command=self.log_on_enter_username)
        self.pw = C.create_entry(self.frame_login_register, 7, 52, 40, 50, fg='black',
                                 command=self.on_password, input_type='password')
        self.login_ok = C.create_button(self.frame_login_register, 10, 15, 75, 65, text='OK',
                                        command=lambda x: self.login_ok_clicked())

    def on_password(self, w, val):
        print("password: " + str(val))
        self.login_info['pw1'] = val


    def login_ok_clicked(self):

        print(f'Ok clicked on Login Button')
        self.frame_login_register.empty()

        df = pd.read_csv('user_registration_info.csv')
        df.drop('Unnamed: 0', inplace=True, axis=1)

        print(f"Login username: {self.login_info['username']}")  # Aru
        print(f"Login pw: {self.login_info['pw1']}")             # 22

        x = df.loc[df.username == self.login_info['username']]
        y = df.loc[df.pw1 == self.login_info['pw1']]
        print(f'X\n{x}')
        print(f'Y\n{y}')
        if x.empty or y.empty:
            C.create_label(self.frame_login_register, 10, 75, 20, 35,
                           text='No Match.', bg='azure')
        else:
            C.create_label(self.frame_login_register, 10, 75, 20, 35,
                           text=f"Logged In.", bg='azure')
            user = self.login_info['username']

            U = User(user)

            self.lgt = C.create_label(self.frame_login_register, 10, 75, 20, 35,
                           text=f"Session: {U.get_name()}", bg='azure')

            self.logout_btn = C.create_button(self.window, 3, 7, 92, 1, text='Logout',
                                             command=lambda x: self.logout_clicked(),
                                             bg='lightgreen')
            self.clear_directory(emitter=True)

            self.upl = C.create_uploader(self.frame_left, 10, 30, 2, 4, filename='./Input PDF/',
                                    command_succ=self.fileupload_successful,
                                    command_fail=self.fileupload_failed)

            self.btn_analyze = C.create_button(self.frame_left, 15, 30, 2, 28, bg='cornflowerblue',
                                               command=lambda x: self.run_analyzer(), text='ANALYZE')

            self.dropdn = C.create_dropdown(self.frame_left, self.bank_list, 15, 65, 35, 4,
                                            bg='powderblue', fg='white', command=self.drop_down_changed)



    def logout_clicked(self):

        self.frame_left.empty()
        self.frame_right.empty()
        self.frame_right_2.empty()
        self.frame_footer_left.empty()
        self.frame_filter.empty()

        self.login_btn = C.create_button(self.window, 3, 7, 92, 1, text='Login',
                                         command=lambda x: self.login_clicked())
        self.register_btn = C.create_button(self.window, 3, 7, 92, 6, text='Register',
                                            command=lambda x: self.register_clicked())
        with self.update_lock:
            self.lgt.set_text("")

        self.clear_directory(emitter=False)      # Removes all user data on logoff.



    def register_clicked(self):

        print(f'Register Clicked')
        self.frame_login_register.empty()
        self.lbl_reg_username = C.create_label(self.frame_login_register, 7, 40, 5, 40, text='Username:', bg='azure')
        self.lbl_reg_pw = C.create_label(self.frame_login_register, 7, 40, 5, 50, text='Password:', bg='azure')
        self.lbl_cnf_pw = C.create_label(self.frame_login_register, 7, 40, 5, 60, text='Confirm Password:', bg='azure')
        self.username = C.create_entry(self.frame_login_register, 7, 52, 40, 40, fg='black',
                                       command=self.reg_on_enter_username)
        self.pw1 = C.create_entry(self.frame_login_register, 7, 52, 40, 50, fg='black',
                                  command=self.reg_on_enter_pw1, input_type='password')
        self.pw2 = C.create_entry(self.frame_login_register, 7, 52, 40, 60, fg='black',
                                  command=self.reg_on_enter_pw2, input_type='password')
        self.login_ok = C.create_button(self.frame_login_register, 10, 15, 75, 75, text='OK',
                                        command=lambda x: self.register_ok_clicked())



    def reg_on_enter_username(self, w, val):
        self.registry_info['username'] = val
        print(val)

    def reg_on_enter_pw1(self, w, val):
        self.registry_info['pw1'] = val

    def reg_on_enter_pw2(self, w, val):
        self.registry_info['pw2'] = val

    def log_on_enter_username(self, w, val):
        self.login_info['username'] = val

    def log_on_enter_pw(self, w, val):
        self.login_info['pw1'] = val



    def register_ok_clicked(self):

        print(f'Ok clicked on Register Button')
        self.frame_login_register.empty()

        # Checking if the user already exists in the records
        try:
            df = pd.read_csv('user_registration_info.csv')          # Read the User records file
            df.drop('Unnamed: 0', inplace=True, axis=1)

            for x in df.username:
                if self.registry_info['username'] in x:
                    lbl = C.create_label(self.frame_login_register, 20, 75, 20, 35,
                                         text='User Record Exists.', bg='azure')
                    return
                else:
                    if self.registry_info['pw1'] != self.registry_info['pw2']:
                        lbl1 = C.create_label(self.frame_login_register, 20, 75, 20, 35,
                                              text='Passwords donot match.', bg='azure')
                        return
        except Exception as e:
            print(f'Exception: {e}. User Record File not existing.')

        C.create_label(self.frame_login_register, 10, 75, 20, 35,
                               text='Registration Successful.', bg='azure')

        # Get and store the user registration information to a text file for now, later pickle it.
        ftwo = {k: self.registry_info[k] for k in list(self.registry_info)[:2]}
        self.master_user = self.master_user.append(ftwo, ignore_index=True)

        if not os.path.isfile('user_registration_info.csv'):
            self.master_user.to_csv('user_registration_info.csv', mode='a',  header='column_names')
        else:
            self.master_user.to_csv('user_registration_info.csv', mode='a', header=False)



    def drop_down_changed(self, w, drpvalue):
        self.notif_1.set_text('Bank: ' + drpvalue)
        self.selected_bank.append(drpvalue)



    def fileupload_successful(self, w, filename):
        self.set_notification(f'{filename} uploaded.')
        self.filename = filename
        print(f'File that was uploaded: \033[0;34m{self.filename}\033[0m')
        return filename



    def fileupload_failed(self, w, filename):
        # lbl_fail = C.create_label(self.frame_left, 5,60,40,1, text=f'{filename} Upload Failed')
        # lbl_fail.css_background_color = self.frame_left_color
        self.set_notification(f'{filename} upload failed.')
        self.filename = filename



    def set_notification(self, text, bar=1):
        if bar == 1:
            self.notif_1.set_text(text)
        if bar == 2:
            self.notif_2.set_text(text)



    def run_analyzer(self):
        self.T = Thread(target=self.run_analysis, daemon=False)
        self.T.start()



    def run_analysis(self):
        '''

        The big function that converts the pdf into tabular data,
        runs an pytesseract OCR on the first page, the others are
        parsed from Tabula. The extracted output is run through
        a postprocessing module then fed into the saved neural
        network model to run classification on the text data and
        return outputs to the UI in form of tables, and other
        interactable output formats.

        '''

        try:
            if self.selected_bank[-1] == 'Axis Bank' or self.selected_bank[-1] == 'Kotak Mahindra Bank'\
                    or self.selected_bank[-1] == 'HDFC Bank':

                with self.update_lock:
                    self.progress.set_value(10)
                    self.set_notification('Initializing. Please wait...', bar=2)
                    import time
                time.sleep(1)

                testpdf = f'Input PDF\\{self.filename}'
                print(f'File being processed: \033[0;34m{testpdf}\033[0m')
                model_name = 'Model Trainer\\Saved Models\\model_ann_99.h5'
                cv_name = 'Model Trainer\\Saved Models\\vectorizer.sav'
                le_name = 'Model Trainer\\Saved Models\\target_label_encoder.sav'

                from time import time
                t1 = time()

                with self.update_lock:
                    self.progress.set_value(25)
                    self.set_notification('Digitizing Document & OCR Extraction...', bar=2)
                G = GetBankData(testpdf)
                G.clear_directory()
                img = G.convert_pdf_image()
                name = G.get_bank_name()

                with self.update_lock:
                    self.progress.set_value(50)
                    self.set_notification('Running Neural Network...', bar=2)
                dx = G.analyze_statement_format()

                if G.bank_name == 'Axis Bank':
                    dxx = G.preprocess_ingest_axis_pdf()
                elif G.bank_name == 'Kotak Mahindra Bank':
                    dxx = G.preprocess_ingest_kotak_pdf()
                elif G.bank_name == 'HDFC Bank':
                    dxx = G.preprocess_ingest_hdfc_pdf()
                else:
                    print('Bank Not Supported')

                dp = G.preprocess_overall_bankdata()
                G.clear_directory(allfiles=False)
                print(f'\033[0;32mOutput file created in {round(time() - t1, 1)}s.\033[0m')

                with self.update_lock:
                    self.progress.set_value(90)
                    self.set_notification('Rendering results...', bar=2)
                self.dt, self.df = load_test_data(model_name, cv_name, le_name)   # Loading master_final as default arg
                self.dc = self.df

                self.table = C.create_table(self.frame_right, self.dt, 91, 97, 2, 4,
                                            align='left', justify='left',
                                            display='block')

                self.btn_graph = C.create_button(self.frame_left, 15, 30, 35, 28, bg='yellowgreen',
                                                 command=lambda x: self.clicked_view_expenses(), text='VIEW EXPENSES')
                with self.update_lock:
                    self.progress.set_value(0)
                    self.set_notification('DONE', bar=2)

                self.btn_analyze = C.create_button(self.frame_left, 15, 30, 68, 28, bg='cornflowerblue',
                                                   text='ANALYTICS', command=lambda x: self.clicked_analytics())

            else:
                self.set_notification('Sorry. This Bank is currently not supported.')

        except IndexError:
            self.set_notification('Please Select Bank from the Dropdown list.')




    def clicked_view_expenses(self):

        self.frame_right.empty()
        self.frame_right_2.empty()

        C.create_table(self.frame_right, self.dt, 91, 97, 2, 4,
                                    align='left', justify='left',
                                    display='block')

        items = self.df.PRED_CAT.unique().tolist()
        print(f'Items in self.df.PRED_CAT unique:\n {items}')

        lblv = C.create_label(self.frame_filter, 5, 100, 0, 3, text=' >>  Filter by :', bg='khaki')
        self.listview = C.create_listview(self.frame_filter, items, 80, 60, 2, 10, bg='whitesmoke')
        self.listview.onselection.do(self.list_view_on_selected)

        self.df.DR = self.df.DR.astype(float)
        self.dr = self.df[self.df.DR > 0]
        print(f'self.dr:\n {self.dr}')

        ff = self.dr.groupby('PRED_CAT').sum()['DR']

        fig = plt.subplots(figsize=(15, 9))
        sns.barplot(ff.index, ff.values, palette='Greens')
        sns.set(font_scale=1.2)
        plt.xlabel('CATEGORY')
        plt.ylabel('Amount')
        plt.title('EXPENSES BY CATEGORY')
        plt.savefig('resx/expenses.png', bbox_inches='tight', pad_inches=0.05, dpi=300)





    def list_view_on_selected(self, w, selected_item_key):

        self.listsel = self.listview.children[selected_item_key].get_text()
        print(f'Selected Item in listview: {self.listsel}, type: {type(self.listsel)}')

        ct = self.dr[self.dr.PRED_CAT == self.listsel]

        # Creates dataframe of the selected entity from the list
        xt = ct.copy()
        ct.DR = ct.DR.astype(str)
        ct.CR = ct.CR.astype(str)
        ct = ct.T

        dr_sum, cr_sum = sum(xt.DR), sum(xt.CR)

        lr = C.create_label(self.frame_right_2, 5, 95, 2, 95, text=f'Total Debit Amount: {round(dr_sum,0)}',
                            bg='lightpink', align='right', justify='right')

        res = []
        for column in ct.columns:
            li = ct[column].tolist()
            res.append(li)

        res.insert(0, ['DATE', 'PARTICULARS', 'DR', 'CR', 'TYPE', 'PREDICTED CATEGORY'])
        self.table2 = C.create_table(self.frame_right_2, res, 80, 97, 2, 4,
                                     align='center', justify='center', display='block')
        self.table2.style['overflow'] = 'overflow'
        self.frame_right_2.append(self.table2)





    def clicked_analytics(self):
        '''
        This will create the listview for Analytics button.
        '''

        self.create_additional_graph()
        self.frame_right_2.empty()
        self.frame_filter.empty()
        self.frame_right_2.css_background_color = 'white'

        try:
            self.img1 = tk.Image(tk.load_resource("./resx/expenses.png"), width="100%", height="50%")
        except Exception as e:
            print(f'\033[0;31mIn Exception - clicked analytics:\033[0m {e}; Expenses Graph missing')

        self.img2 = tk.Image(tk.load_resource("./resx/expenses_type_additional.png"),
                             width="100%", height="50%", top="50%")

        self.frame_right_2.append(self.img1)
        self.frame_right_2.append(self.img2)

        items = self.df.TYPE.unique().tolist() + self.df.PRED_CAT.unique().tolist()     # Adding Pred_Cat for analytics

        lblv = C.create_label(self.frame_filter, 5, 100, 0, 3, text='>>  Filter by:', bg='khaki')
        self.listview_2 = C.create_listview(self.frame_filter, items, 80, 60, 2, 10, bg='whitesmoke')
        self.listview_2.onselection.do(self.list_view_on_selected_2)




    def list_view_on_selected_2(self, w, selected_item_key_2):
        '''
        This will create the list view for analytics.
        '''

        self.key1 = selected_item_key_2
        self.frame_right_2.empty()
        self.listsel_2 = self.listview_2.children[selected_item_key_2].get_text()
        self.key2 = self.listsel_2

        print(f'Selected Item in listview_2: {self.listsel_2}, type: {type(self.listsel_2)}')

        if self.listsel_2.isupper():                        # The only way to discern between type and pred in listview.
            ct = self.df[self.df.TYPE == self.listsel_2]
        else:
            ct = self.df[self.df.PRED_CAT == self.listsel_2]


        self.dct = ct
        # Creates dataframe of the selected entity from the list
        xt = ct.copy()
        ct.DR = ct.DR.astype(str)
        ct.CR = ct.CR.astype(str)
        ct = ct.T
        dr_sum_2, cr_sum_2 = sum(xt.DR), sum(xt.CR)

        lr = C.create_label(self.frame_right_2, 5, 95, 2, 95, text=f'Total Debit Amount: {round(dr_sum_2,0)}',
                            bg='lightpink', align='right', justify='right')
        print(f'ct dataframe from list_selection_2:\n{xt.head()}')


        res2 = []
        for column in ct.columns:
            li = ct[column].tolist()
            res2.append(li)

        res2.insert(0, ['DATE', 'PARTICULARS', 'DR', 'CR', 'TYPE', 'PREDICTED CATEGORY'])
        self.table3 = C.create_table(self.frame_right_2, res2, 80, 97, 2, 4,
                                     align='center', justify='center', display='block')
        self.table3.style['overflow'] = 'overflow'

        self.T.join()
        self.frame_right.empty()
        self.graph_by_filter_2()



    def graph_by_filter_2(self):
        '''
        This creates the graph after the user clicks on the 'Analytics'
        button on the UI and then moves through the listview_2 to
        cycle through the created graphs.

        '''
        palette = ['PuOr', 'Reds', 'Greens', 'coolwarm', 'autumn']

        if self.listsel_2.isupper():
            dk = self.dct[self.dct.TYPE == self.key2]
            ch = random.choice(palette)
            fig = plt.subplots(figsize=(15, 9))
            sns.countplot(dk.PRED_CAT, label='HEY', palette=ch)
            sns.set(font_scale=1.4)
            plt.xlabel('Transaction Type')
            plt.ylabel('Count')
            plt.title(f'Transactions bY {self.key2}')
            plt.savefig(r'resx/anomaly.jpg', pad_inches=0.1, dpi=200)


        else:
            dk = self.df[self.df.PRED_CAT == self.key2]
            ch = random.choice(palette)
            fig = plt.subplots(figsize=(15, 9))
            sns.lineplot(range(len(dk)), dk.DR, palette=ch)
            sns.set(font_scale=1.4)
            plt.xlabel('Frequency of Transaction')
            plt.ylabel('Amount')
            plt.title(f'Expenses of {self.key2} over Times of Transactions')
            plt.savefig(r'resx/anomaly.jpg', pad_inches=0.1, dpi=200)

        self.img2 = tk.Image(tk.load_resource("./resx/anomaly.jpg"), width="96%", height="75%", margin='2%')
        self.frame_right.append(self.img2)




    def create_additional_graph(self):
        '''
        This function creates an additional graph and adds to
        the UI when the user clicks on Analytics after clicking
        'Viewing Expenses'. If 'View expenses' was not clicked,
        only the graph created by this function wil be visible.

        '''

        plt.subplots(figsize=(15, 9))
        sns.countplot(self.dc.TYPE, palette='Blues')
        sns.set(font_scale=1.2)
        plt.xlabel('Transaction Type')
        plt.ylabel('Count')
        plt.title(f'TRANSACTIONS BY TYPES')

        plt.savefig(r'resx/expenses_type_additional.png', bbox_inches='tight', pad_inches=0.05)
        plt.clf()



configuration = {'config_project_name': 'MainScreen',
                 'config_address': '127.0.0.1',
                 'config_port': 8081, 'config_multiple_instance': True,
                 'config_enable_file_cache': True,
                 'config_start_browser': True,
                 'config_resourcepath': './resx/'}


start(BankStatementAnalyzer, address=configuration['config_address'], port=configuration['config_port'],
      multiple_instance=configuration['config_multiple_instance'],
      enable_file_cache=configuration['config_enable_file_cache'],
      start_browser=configuration['config_start_browser'])

