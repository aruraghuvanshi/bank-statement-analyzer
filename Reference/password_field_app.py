"""
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import remi.gui as gui
from remi import start, App


class MyApp(App):
    def __init__(self, *args):
        super(MyApp, self).__init__(*args)

    def main(self):
        #margin 0px auto allows to center the app to the screen
        wid = gui.VBox(width=300, height=200, margin='0px auto')
        lbl = gui.Label('Password', width='80%', height='50%')
        lbl.style['margin'] = 'auto'

        txt = gui.Input(input_type='password', width=200, height=30)
        txt.style['background-color'] = 'lightgray'
        txt.attributes['type'] = 'password'
        txt.onchange.connect(self.on_password)

        # appending a widget to another, the first argument is a string key
        wid.append(lbl)
        wid.append(txt)

        # returning the root widget
        return wid

    def on_password(self, emitter, new_value):
        print("password: " + str(new_value))


if __name__ == "__main__":
    start(MyApp, debug=True)
