from remi import App, start
import os
import remi.gui as tk
from creators import C
import datetime
# from axis_bank_statement_analyzer_trainer_v2 import load_test_data, ingest_test_pdf
from getbankdata import GetBankData
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from user import User
from threading import Thread
from run_saved_model import load_test_data
'''
USE MAIN.PY for running.
THIS IS WIP

+ Retrain Model
- Swap DR with CR in Kotak
- Check Dropdown of Banklist feasibility and add Kotak
+ Modify code to add new frame for filtering options 
+ Add Filters to view graphs on transaction TYPES and PRED_CAT
+ Use the choosefile to choose file instead of hardcoded path

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
        self.dr = pd.DataFrame()
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
        self.frame_header_color = 'deepskyblue'
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
        self.frame_header = C.create_container(self.window, 10, 90, 0, 0)
        self.frame_header.css_background_color = self.frame_header_color
        self.frame_header.css_top = "0%"

        self.frame_footer_left = C.create_container(self.window, 12, 20, 0, 87)
        self.frame_footer_left.css_background_color = self.frame_footer_left_color

        self.progress = C.create_progress(self.window, 1, 100, 0, 99, a=0, b=100)

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

        # --------------------- LABELS ---------------------------------------------------------- ]

        lbl_header = C.create_label(self.frame_header, 20, 25, 10, 30, text='BANK STATEMENT ANALYZER',
                                    bg='deepskyblue', fg='white')
        lbl_header.css_font_size = '18px'

        lbl_subheader = C.create_label(self.frame_header, 10, 20, 13.35, 60,
                                       text='-- Aru Raghuvanshi build 07042021',
                                       bg='deepskyblue', fg='white')
        lbl_subheader.css_font_size = '12px'

        lbl_datetime = C.create_label(self.frame_header, 20, 7, 93, 40, text=f'Date: {self.date}',
                                      bg='deepskyblue', fg='white', align='right')
        lbl_datetime.css_font_size = '14px'

        self.notif_1 = C.create_label(self.frame_footer_left, 6, 100, 0, 10, text='')
        self.notif_2 = C.create_label(self.frame_footer_left, 6, 100, 0, 60, text='')

        # --------------------- APPENDS --------------------------------------------------------- ]
        # self.window.append(self.frame_right)
        # self.window.append(self.frame_right_2)
        # self.window.append(self.frame_left)
        # self.window.append(self.frame_header)
        # self.window.append(self.frame_footer_left)

        self.login_btn = C.create_button(self.window, 3, 7, 92, 1, text='Login',
                                         command=lambda x: self.login_clicked())
        self.register_btn = C.create_button(self.window, 3, 7, 92, 6, text='Register',
                                            command=lambda x: self.register_clicked())

        return self.window

    # ====================== FUNCTIONS ============================================================ ]

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

        # Do the username/password match here
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

            # --------------------- FILE UPLOADER & SELECTOR -------------------------------------- ]
            upl = C.create_uploader(self.frame_left, 10, 30, 2, 4, filename='./files/')
            upl.onsuccess.do(self.fileupload_successful)
            upl.onfailed.do(self.fileupload_failed)

            # --------------------- BUTTONS --------------------------------------------------------- ]
            self.btn_analyze = C.create_button(self.frame_left, 15, 30, 2, 28, bg='cornflowerblue',
                                               command=lambda x: self.run_analyzer(), text='ANALYZE')

            # --------------------- DROPDOWNS --------------------------------------------------------- ]
            self.dropdn = C.create_dropdown(self.frame_left, self.bank_list, 15, 65, 35, 4,
                                            bg='powderblue', fg='white', command=self.drop_down_changed)


    def logout_clicked(self):

        self.frame_left.empty()
        self.frame_right.empty()
        self.frame_right_2.empty()
        self.frame_footer_left.empty()
        self.login_btn = C.create_button(self.window, 3, 7, 92, 1, text='Login',
                                         command=lambda x: self.login_clicked())
        self.register_btn = C.create_button(self.window, 3, 7, 92, 6, text='Register',
                                            command=lambda x: self.register_clicked())
        with self.update_lock:
            self.lgt.set_text("")


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
        except:
            print(f'User Record File not existing.')

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
        lbl_succ = C.create_label(self.frame_left, 7, 98, 2, 15, text=f'{filename} uploaded.',
                                  display='flex-end', align='center', justify='space-around')
        lbl_succ.css_background_color = 'lightgreen'
        return filename


    def fileupload_failed(self, w, filename):
        lbl_fail = C.create_label(self.frame_left, 5,60,40,1, text=f'{filename} Upload Failed')
        lbl_fail.css_background_color = self.frame_left_color


    def set_notification(self, text, bar=1):
        if bar == 1:
            self.notif_1.set_text(text)
        if bar == 2:
            self.notif_2.set_text(text)


    def compute_progress(self, val):
        self.progress.set_value(val)


    def run_analyzer(self):
        self.T = Thread(target=self.run_analysis, daemon=False)
        self.T.start()
        # self.run_analysis()

    def run_analysis(self):

        try:
            if self.selected_bank[-1] == 'Axis Bank' or self.selected_bank[-1] == 'Kotak Mahindra Bank':

                with self.update_lock:
                    self.progress.set_value(10)
                    self.set_notification('Initializing. Please wait...', bar=2)
                    import time
                time.sleep(1)

                # testpdf = r'Resources/aru_kotak_bank.pdf'
                testpdf = r'Resources/axis_test_bank.pdf'
                model_name = 'models\\model_ann_98.h5'
                cv_name = 'models\\vectorizer.sav'
                le_name = 'models\\label_encoder.sav'

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
                else:
                    print('Bank Not Supported')

                dp = G.preprocess_overall_bankdata()
                G.clear_directory(allfiles=False)
                print(f'\033[0;32mOutput file created in {round(time() - t1, 1)}s.\033[0m')

                with self.update_lock:
                    self.progress.set_value(90)
                    self.set_notification('Rendering results...', bar=2)
                self.dt, self.df = load_test_data(model_name, cv_name, le_name)    # Loading master_final as default arg

                self.table = C.create_table(self.frame_right, self.dt, 91, 97, 2, 4,
                                            align='left', justify='left',
                                            display='block')

                self.btn_graph = C.create_button(self.frame_left, 15, 30, 35, 28, bg='yellowgreen',
                                                 command=lambda x: self.create_graph(), text='VIEW EXPENSES')
                with self.update_lock:
                    self.progress.set_value(0)
                    self.set_notification('DONE', bar=2)

                self.btn_analyze = C.create_button(self.frame_left, 15, 30, 68, 28, bg='cornflowerblue',
                                                   text='ANALYTICS', command=lambda x: self.clicked_analytics())

            else:
                self.set_notification('Sorry. This Bank is currently not supported.')

        except IndexError:
            self.set_notification('Please Select Bank from the Dropdown list.')



    def create_graph(self):
        self.frame_right.empty()

        C.create_table(self.frame_right, self.dt, 91, 97, 2, 4,
                                    align='left', justify='left',
                                    display='block')

        def expense_by_category(df, cat='PRED_CAT', exception=False,
                                exceptvalue='Woodstock', exceptvalue2='Credit'):

            print(f'self.df:\n{self.df}')
            if exception:
                print('in exception to filter exception values')
                self.dr = df[(df[cat] != exceptvalue) & (df[cat] != exceptvalue2)]
            else:
                self.dr = df[df.DR > 0]
                print('in exception else (normal)')

            catdf = self.dr.groupby(cat).sum()['DR'].plot(kind='bar', figsize=(15, 10),
                                                          color='yellowgreen', fontsize=14,
                                                          title='Expenses by Category')
            catdf.set_xlabel('Category of Expense', fontsize=20)
            catdf.set_ylabel('Amount in Rupees', fontsize=20)
            catdf.set_title('Expenses by Category', fontsize=20)
            print(f'catdf: \n {catdf}')

            # with self.update_lock:
            #     plt.show()                          #ValueError: signal only works in main thread
            plt.savefig('resx/expenses.png')


        items = self.df.PRED_CAT.unique().tolist()
        print(f'df.PRED_CAT items: {items}')

        lblv = C.create_label(self.frame_filter, 5, 100, 0, 3, text='>>  Filter by:', bg='khaki')
        self.listview = C.create_listview(self.frame_filter, items, 80, 60, 2, 10, bg='whitesmoke')
        self.listview.onselection.do(self.list_view_on_selected)

        expense_by_category(self.df, cat='PRED_CAT', exception=False,
                            exceptvalue='Woodstock', exceptvalue2='Credit')



    def list_view_on_selected(self, w, selected_item_key):
        """ The selection event of the listView, returns a key of the clicked event.
            You can retrieve the item rapidly """


        self.listsel = self.listview.children[selected_item_key].get_text()
        print(f'Selected Item in listview: {self.listsel}, type: {type(self.listsel)}')

        ct = self.dr[self.dr.PRED_CAT == self.listsel]

        # Creates dataframe of the selected entity from the list
        xt = ct.copy()
        ct.DR = ct.DR.astype(str)
        ct.CR = ct.CR.astype(str)
        ct = ct.T

        dr_sum, cr_sum = sum(xt.DR), sum(xt.CR)

        lr = C.create_label(self.frame_right_2, 5, 95, 2, 95, text=f'Total Amount: {dr_sum}',
                            bg='lightpink', align='right', justify='right')
        print(f'ct dataframe from list selection:\n{xt}')
        res = []
        for column in ct.columns:
            li = ct[column].tolist()
            res.append(li)
        print(f'res list: {res}')
        res.insert(0, ['DATE', 'PARTICULARS', 'DR', 'CR', 'TYPE', 'PREDICTED CATEGORY'])
        self.table2 = C.create_table(self.frame_right_2, res, 80, 97, 2, 4,
                                     align='center', justify='center', display='block')
        self.table2.style['overflow'] = 'overflow'
        self.frame_right_2.append(self.table2)


    def clicked_analytics(self):

        '''
        This will create the list for Analytics button.

        '''

        self.frame_right_2.empty()
        self.frame_filter.empty()
        C.create_image(self.frame_right_2, '/path:expenses.png', 50, 96, 2, 2)
        C.create_image(self.frame_right_2, '/path:expenses_type.png', 50, 96, 2, 55)

        print(f'self.df in clicked analytics:\n {self.df}')
        items = self.df.TYPE.unique().tolist()
        print(f'df.TYPE items: {items}')

        lblv = C.create_label(self.frame_filter, 5, 100, 0, 3, text='>>  Filter by:', bg='khaki')
        self.listview_2 = C.create_listview(self.frame_filter, items, 80, 60, 2, 10, bg='whitesmoke')
        self.listview_2.onselection.do(self.list_view_on_selected_2)


    def list_view_on_selected_2(self, w, selected_item_key_2):

        '''
        This will create the list view for analytics.
        '''

        def expense_by_type(df, cat='TYPE'): #, exception=False,
                                # exceptvalue='Woodstock', exceptvalue2='Credit'):

            print(f'self.df:\n{self.df}')

            dk = df[df.DR > 0]
            print(f'dk: \n {dk}')

            typedf = dk.groupby(cat).sum()['DR'].plot(kind='bar', figsize=(15, 10),
                                                          color='lightskyblue', fontsize=14,
                                                          title='Expenses by Type')
            typedf.set_xlabel('Category of Expense', fontsize=20)
            typedf.set_ylabel('Amount in Rupees', fontsize=20)
            typedf.set_title('Expenses by Type', fontsize=20)
            print(f'typedf: \n {typedf}')

            # with self.update_lock:
            #     plt.show()                          #ValueError: signal only works in main thread
            plt.savefig('resx/expenses_type.png')

        self.frame_right_2.empty()
        self.listsel_2 = self.listview_2.children[selected_item_key_2].get_text()
        print(f'Selected Item in listview_2: {self.listsel_2}, type: {type(self.listsel_2)}')

        ct = self.df[self.df.TYPE == self.listsel_2]

        # Creates dataframe of the selected entity from the list
        xt = ct.copy()
        ct.DR = ct.DR.astype(str)
        ct.CR = ct.CR.astype(str)
        ct = ct.T

        dr_sum_2, cr_sum_2 = sum(xt.DR), sum(xt.CR)

        lr = C.create_label(self.frame_right_2, 5, 95, 2, 95, text=f'Total Amount: {dr_sum_2}',
                            bg='lightpink', align='right', justify='right')
        print(f'ct dataframe from list selection:\n{xt}')
        res2 = []
        for column in ct.columns:
            li = ct[column].tolist()
            res2.append(li)
        print(f'res2 list: {res2}')

        res2.insert(0, ['DATE', 'PARTICULARS', 'DR', 'CR', 'TYPE', 'PREDICTED CATEGORY'])
        self.table3 = C.create_table(self.frame_right_2, res2, 80, 97, 2, 4,
                                     align='center', justify='center', display='block')
        self.table3.style['overflow'] = 'overflow'
        self.frame_right_2.append(self.table3)

        self.frame_right.empty()
        expense_by_type(self.df)
        C.create_image(self.frame_right, '/path:expenses_type.png', 60, 96, 2, 2)




configuration = {'config_project_name': 'MainScreen',
                 'config_address': '127.0.0.1',
                 'config_port': 8084, 'config_multiple_instance': True,
                 'config_enable_file_cache': True,
                 'config_start_browser': True,
                 'config_resourcepath': './resx/'}


start(BankStatementAnalyzer, address=configuration['config_address'], port=configuration['config_port'],
      multiple_instance=configuration['config_multiple_instance'],
      enable_file_cache=configuration['config_enable_file_cache'],
      start_browser=configuration['config_start_browser'])

