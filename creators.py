import remi.gui as tk


class Creator:

    def create_label(self, frame, H, W, L, T, text='.', bg='white', fg='black',
                     align='center', justify='space-around', display='inline',
                     position='absolute', px=False):

        lbl = tk.Label(text=text)
        lbl.variable_name = 'lbl'
        if px:
            lbl.css_height = str(f'{H}px')
            lbl.css_width = str(f'{W}px')
            lbl.css_left = str(f'{L}px')
            lbl.css_top = str(f'{T}px')
        else:
            lbl.css_height = str(f'{H}%')
            lbl.css_width = str(f'{W}%')
            lbl.css_left = str(f'{L}%')
            lbl.css_top = str(f'{T}%')
        lbl.css_background_color = bg
        lbl.css_color = fg
        lbl.css_align_self = align
        lbl.css_align_content = align
        lbl.css_align_items = align
        lbl.css_display = display
        lbl.css_position = position
        lbl.css_justify_content = justify
        frame.append(lbl)
        return lbl


    def create_button(self, frame, H, W, L, T, command=None, text='.', bg='navyblue', fg='white',
                      align='center', justify='space-around',
                      display='inline', position='absolute', px=False):

        btn = tk.Button(text=text)
        if px:
            btn.variable_name = 'btn'
            btn.css_height = str(f'{H}px')
            btn.css_width = str(f'{W}px')
            btn.css_left = str(f'{L}px')
            btn.css_top = str(f'{T}px')
        else:
            btn.variable_name = 'btn'
            btn.css_height = str(f'{H}%')
            btn.css_width = str(f'{W}%')
            btn.css_left = str(f'{L}%')
            btn.css_top = str(f'{T}%')
        btn.css_background_color = bg
        btn.css_color = fg
        btn.css_align_self = align
        btn.css_align_content = align
        btn.css_align_items = align
        btn.css_display = display
        btn.css_justify_content = align
        btn.css_position = position
        btn.css_justify_content = justify
        btn.onclick.do(command)
        frame.append(btn)
        return btn


    def create_uploader(self, frame, H, W, L, T, filename=None, bg='navyblue', fg='white',
                      align='center', justify='space-around',
                      display='inline', position='absolute', px=False):

        btn = tk.FileUploader(savepath=filename)
        if px:
            btn.variable_name = 'upl'
            btn.css_height = str(f'{H}px')
            btn.css_width = str(f'{W}px')
            btn.css_left = str(f'{L}px')
            btn.css_top = str(f'{T}px')
        else:
            btn.variable_name = 'btn'
            btn.css_height = str(f'{H}%')
            btn.css_width = str(f'{W}%')
            btn.css_left = str(f'{L}%')
            btn.css_top = str(f'{T}%')
        btn.css_background_color = bg
        btn.css_color = fg
        btn.css_align_self = align
        btn.css_align_content = align
        btn.css_align_items = align
        btn.css_display = display
        btn.css_justify_content = align
        btn.css_position = position
        btn.css_justify_content = justify
        frame.append(btn)
        return btn



    def create_container(self, frame, H, W, L, T, bg='ivory', fg='black',
                      align='center', justify='space-around',
                      display='inline', position='absolute', px=False):

        btn = tk.Container()
        if px:
            btn.variable_name = 'ctn'
            btn.css_height = str(f'{H}px')
            btn.css_width = str(f'{W}px')
            btn.css_left = str(f'{L}px')
            btn.css_top = str(f'{T}px')
        else:
            btn.variable_name = 'btn'
            btn.css_height = str(f'{H}%')
            btn.css_width = str(f'{W}%')
            btn.css_left = str(f'{L}%')
            btn.css_top = str(f'{T}%')
        btn.css_background_color = bg
        btn.css_color = fg
        btn.css_align_self = align
        btn.css_align_content = align
        btn.css_align_items = align
        btn.css_display = display
        btn.css_justify_content = align
        btn.css_position = position
        btn.css_justify_content = justify
        frame.append(btn)
        return btn



    def create_table(self, frame, lst, H, W, L, T, bg='seashell', fg='black',
                      align='center', justify='space-around', font_size='12px',
                      display='inline', position='absolute', px=False):

        btn = tk.Table.new_from_list(content=lst)
        if px:
            btn.variable_name = 'tbl'
            btn.css_height = str(f'{H}px')
            btn.css_width = str(f'{W}px')
            btn.css_left = str(f'{L}px')
            btn.css_top = str(f'{T}px')
        else:
            btn.variable_name = 'btn'
            btn.css_height = str(f'{H}%')
            btn.css_width = str(f'{W}%')
            btn.css_left = str(f'{L}%')
            btn.css_top = str(f'{T}%')
        btn.css_font_size = font_size
        btn.css_background_color = bg
        btn.css_color = fg
        btn.css_align_self = align
        btn.css_align_content = align
        btn.css_align_items = align
        btn.css_display = display
        btn.css_justify_content = align
        btn.css_position = position
        btn.css_justify_content = justify
        frame.append(btn)
        return btn


    def create_image(self, frame, imagepath, H, W, L, T, bg='navyblue', fg='white',
                      align='center', justify='space-around',
                      display='inline', position='absolute', px=False):

        btn = tk.Image(image=imagepath)
        if px:
            btn.variable_name = 'img'
            btn.css_height = str(f'{H}px')
            btn.css_width = str(f'{W}px')
            btn.css_left = str(f'{L}px')
            btn.css_top = str(f'{T}px')
        else:
            btn.variable_name = 'btn'
            btn.css_height = str(f'{H}%')
            btn.css_width = str(f'{W}%')
            btn.css_left = str(f'{L}%')
            btn.css_top = str(f'{T}%')
        btn.css_background_color = bg
        btn.css_color = fg
        btn.css_align_self = align
        btn.css_align_content = align
        btn.css_align_items = align
        btn.css_display = display
        btn.css_justify_content = align
        btn.css_position = position
        btn.css_justify_content = justify
        frame.append(btn)
        return btn


    def create_listview(self, frame, lst, H, W, L, T, command=None, bg='skyblue', fg='black',
                      align='center', justify='space-around',
                      display='inline', position='absolute', px=False):

        btn = tk.ListView.new_from_list(items=lst)
        if px:
            btn.variable_name = 'lvw'
            btn.css_height = str(f'{H}px')
            btn.css_width = str(f'{W}px')
            btn.css_left = str(f'{L}px')
            btn.css_top = str(f'{T}px')
        else:
            btn.variable_name = 'btn'
            btn.css_height = str(f'{H}%')
            btn.css_width = str(f'{W}%')
            btn.css_left = str(f'{L}%')
            btn.css_top = str(f'{T}%')

        btn.css_background_color = bg
        btn.css_color = fg
        btn.css_align_self = align
        btn.css_align_content = align
        btn.css_align_items = align
        btn.css_display = display
        btn.css_justify_content = align
        btn.css_position = position
        btn.css_justify_content = justify
        btn.onselection.do(command)
        frame.append(btn)
        return btn



    def create_progress(self, frame, H, W, L, T, a, b=100, bg='lightgreen', fg='pink',
                      align='center', justify='space-around',
                      display='inline', position='absolute', px=False):

        btn = tk.Progress(a, b)
        if px:
            btn.variable_name = 'prg'
            btn.css_height = str(f'{H}px')
            btn.css_width = str(f'{W}px')
            btn.css_left = str(f'{L}px')
            btn.css_top = str(f'{T}px')
        else:
            btn.variable_name = 'btn'
            btn.css_height = str(f'{H}%')
            btn.css_width = str(f'{W}%')
            btn.css_left = str(f'{L}%')
            btn.css_top = str(f'{T}%')
        btn.css_background_color = bg
        btn.css_color = fg
        btn.css_align_self = align
        btn.css_align_content = align
        btn.css_align_items = align
        btn.css_display = display
        btn.css_justify_content = align
        btn.css_position = position
        btn.css_justify_content = justify
        frame.append(btn)
        return btn


    def create_dropdown(self, frame, lst, H, W, L, T, command=None, bg='navyblue', fg='white',
                      align='center', justify='space-around',
                      display='inline', position='absolute', px=False):

        btn = tk.DropDown.new_from_list(lst)
        if px:
            btn.variable_name = 'drp'
            btn.css_height = str(f'{H}px')
            btn.css_width = str(f'{W}px')
            btn.css_left = str(f'{L}px')
            btn.css_top = str(f'{T}px')
        else:
            btn.variable_name = 'btn'
            btn.css_height = str(f'{H}%')
            btn.css_width = str(f'{W}%')
            btn.css_left = str(f'{L}%')
            btn.css_top = str(f'{T}%')
        btn.css_background_color = bg
        btn.css_color = fg
        btn.css_align_self = align
        btn.css_align_content = align
        btn.css_align_items = align
        btn.css_display = display
        btn.css_justify_content = align
        btn.css_position = position
        btn.css_justify_content = justify
        btn.select_by_value(lst[0])
        btn.onchange.do(command)
        frame.append(btn)
        return btn


    def create_entry(self, frame, H, W, L, T, command=None, bg='navyblue', fg='white',
                      align='center', justify='space-around', input_type='regular',
                      display='inline', position='absolute', px=False):

        '''
         input_type are either 'password', 'regular' or 'text'.
        '''
        if input_type == 'password':
            btn = tk.Input(input_type='password')
            btn.attributes['type'] = 'password'
            btn.style['background-color'] = 'lightgray'
            btn.onchange.connect(command)
        elif input_type == 'text':
            btn = tk.TextInput()
        else:
            btn = tk.Input()
            btn.style['background-color'] = 'lightgray'

        # btn.set_text('')
        if px:
            btn.variable_name = 'drp'
            btn.css_height = str(f'{H}px')
            btn.css_width = str(f'{W}px')
            btn.css_left = str(f'{L}px')
            btn.css_top = str(f'{T}px')
        else:
            btn.variable_name = 'btn'
            btn.css_height = str(f'{H}%')
            btn.css_width = str(f'{W}%')
            btn.css_left = str(f'{L}%')
            btn.css_top = str(f'{T}%')
        btn.css_background_color = bg
        btn.css_color = fg
        btn.css_align_self = align
        btn.css_align_content = align
        btn.css_align_items = align
        btn.css_display = display
        btn.css_justify_content = align
        btn.css_position = position
        btn.css_justify_content = justify
        btn.onchange.do(command)
        frame.append(btn)
        return btn


    def create_inputdialogue(self, frame, H, W, L, T, title='ttl', message= 'desc',
                             command=None, bg='navyblue', fg='white',
                             align='center', justify='space-around',
                             display='inline', position='absolute', px=False):

        btn = tk.InputDialog(title=title, message=message)
        btn.confirm_value.do(command)

        if px:
            btn.variable_name = 'drp'
            btn.css_height = str(f'{H}px')
            btn.css_width = str(f'{W}px')
            btn.css_left = str(f'{L}px')
            btn.css_top = str(f'{T}px')
        else:
            btn.variable_name = 'btn'
            btn.css_height = str(f'{H}%')
            btn.css_width = str(f'{W}%')
            btn.css_left = str(f'{L}%')
            btn.css_top = str(f'{T}%')
        btn.css_background_color = bg
        btn.css_color = fg
        btn.css_align_self = align
        btn.css_align_content = align
        btn.css_align_items = align
        btn.css_display = display
        btn.css_justify_content = align
        btn.css_position = position
        btn.css_justify_content = justify
        frame.append(btn)
        return btn




C = Creator()