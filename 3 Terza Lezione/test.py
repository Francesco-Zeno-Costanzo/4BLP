"""
Code to test the firts exercise code
"""
import io
import sys
import time
import inspect
import unittest
import subprocess
import contextlib
import tracemalloc
import importlib.util

import numpy as np
import tkinter as tk
from tkinter import scrolledtext


# Parameters of function and expected result form the text of the exercise
obs_p = np.array([0, 3, 5, 1, 5, 667, 2, 4, 9, 3, 33])
obs_e = np.array([66.545, 60.109])
primi = np.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,
                  67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,
                  139,149,151,157,163,167,173,179,181,191,193,197,199])

cesar_p = {1:('Hola',13, 0), 2:('Ubyn',13,-1)}
cesar_e = {1:'Ubyn',         2:'Hola'} 
clean_p = np.array([3, 4, np.nan, 8, 1, np.nan, np.nan, 9])
clean_e = np.array([3, 4, 0     , 8, 1, 0     , 0     , 9])
eq2_p   = {1:(0, 1, 1), 2:(1, -2, 1), 3:(1, 1, 2) }
eq2_e   = {1: np.array([-1]), 2: np.array([1, 1]), 3: np.array([])}

test_params = {
    'osservabile'       : {'params': (obs_p,),       'expected': obs_e},
    'area'              : {'params': (2, 4),         'expected': 4},
    'pitagora'          : {'params': ([3, 4, 5], 5), 'expected': 0},
    'pi_greco_for'      : {'params': (int(1e6),),    'expected':np.pi},
    'pi_greco_vec'      : {'params': (int(1e6),),    'expected':np.pi},
    'decimal_to_binary' : {'params': (42,),          'expected':'101010'},
    'binary_to_decimal' : {'params': ('101010',),    'expected':42},
    'trova_primi'       : {'params': (200,),         'expected':primi},
    'palindromo'        : {'params': (12345654321,), 'expected':True},
    'cesare'            : {'params': cesar_p,        'expected':cesar_e},
    'esa_to_dec'        : {'params': ('1A3F',),      'expected':6719},
    'dec_to_esa'        : {'params': (6719,),        'expected':'1A3F'},
    'clean_data'        : {'params': (clean_p,),     'expected':clean_e},
    'eq'                : {'params': eq2_p,          'expected':eq2_e},
    'fattori'           : {'params': (713491741,),   'expected':[389, 719, 2551]},
    'goldbach'          : {'params': (56,),          'expected':[(3, 53), (13, 43), (19, 37)]},
    'newton_sqrt'       : {'params': (2864,),        'expected':53.51635264103861},
    'EMCD'              : {'params': (4653, 2793),   'expected': np.array([3, 464, -773])}

}

class TestEsercizio(unittest.TestCase):
    ''' Class for unit test
    '''

    file_to_test = "" # file of the exercise

    @classmethod
    def setUpClass(cls):
        ''' Class method to load the codo to test
        '''
        # Import module
        module_name = TestEsercizio.file_to_test[:3]
        file_path   = TestEsercizio.file_to_test
        spec        = importlib.util.spec_from_file_location(module_name, file_path)
        cls.module  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_functions(self):
        ''' Test for evry function inside the code
        '''
        # Collect all functions
        functions = [func for name, func in inspect.getmembers(self.module, inspect.isfunction)]
        
        for function in functions:
            with self.subTest(function=function.__name__):
                # Verify the presence of docstring
                self.assertIsNotNone(function.__doc__, f"La funzione {function.__name__} non ha una docstring.")
                
                # Find parameters and result for the actual function
                test_info = test_params.get(function.__name__, {'params': (), 'expected': None})
                params    = test_info['params']
                expected  = test_info['expected']

                if type(params) is dict:
                    for i in range(1, len(params)+1):

                        start_time = time.time()  # for time
                        tracemalloc.start()       # for memory

                        if function.__name__ in test_params:
                            try:
                                print(params[i])
                                result = function(*params[i])
                            except Exception as e:
                                log_results(function.__name__, error=f"La funzione {function.__name__} ha sollevato un'eccezione: {e}")
                                self.fail(f"La funzione {function.__name__} ha sollevato un'eccezione: {e}")

                        # Current memory and peak of memeory used
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()

                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        # Check the output
                        if function.__name__ in test_params:
                            try:
                                msg = f"Output errato per {function.__name__}: per {params[i]} il risultato atteso è {expected[i]}, ma ho ottenuto {result}"
                                try:
                                    n = len(expected[i])
                                    if n > 1:
                                        for res, exp in zip(result, expected[i]):
                                            self.assertAlmostEqual(res, exp, 3, msg) # 3 in decimal place
                                except :
                                    self.assertAlmostEqual(result, expected[i], 3, msg) # 3 is decimal place
                                    
                            except AssertionError as e:
                                log_results(function.__name__, error=str(e))
                                raise

                            # Result on GUI
                            log_results(function.__name__, elapsed_time, current, peak, result)
                else:
                    start_time = time.time()  # for time
                    tracemalloc.start()       # for memory

                    if function.__name__ in test_params:
                        try:
                            result = function(*params)
                        except Exception as e:
                            log_results(function.__name__, error=f"La funzione {function.__name__} ha sollevato un'eccezione: {e}")
                            self.fail(f"La funzione {function.__name__} ha sollevato un'eccezione: {e}")

                    # Current memory and peak of memeory used
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    # Check the output
                    if function.__name__ in test_params:
                        try:
                            msg = f"Output errato per {function.__name__}: per {params} il risultato atteso è {expected}, ma ho ottenuto {result}"
                            try:
                                n = len(expected)
                                if n > 1:
                                    for res, exp in zip(result, expected):
                                        self.assertAlmostEqual(res, exp, 3, msg) # 3 in decimal place
                            except :
                                self.assertAlmostEqual(result, expected, 3, msg) # 3 is decimal place
                                
                        except AssertionError as e:
                            log_results(function.__name__, error=str(e))
                            raise

                        # Result on GUI
                        log_results(function.__name__, elapsed_time, current, peak, result)


    def test_script(self):
        ''' Test for the all script 
        '''
        start_time = time.time()
        tracemalloc.start()

        process = subprocess.Popen(['python3', TestEsercizio.file_to_test], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        elapsed_time = end_time - start_time

        log_results("Script intero", elapsed_time, current, peak, "\n"+stdout.decode(), stderr.decode())
        if process.returncode != 0:
            self.fail(f"Il codice ha sollevato un'eccezione: {stderr.decode()}")

    def test_pylint(self):
        ''' Pylint's test
        '''
        process = subprocess.Popen(['pylint', TestEsercizio.file_to_test], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        log_results("pylint", result=stdout.decode(), error=stderr.decode())


def log_results(function_name, elapsed_time=None, current_memory=None, peak_memory=None, result=None, error=None):
    '''
    Function to log all information on GUI

    Parameter
    ---------
    function_name : string
        name of the function we analyze
    elapsed_time : None or float, optional, default None
        Elapsed time for function execution
    current_memory : None or float, optional, default None
        memory used at the time when get_traced_memory() is called
    peak_memory : None or float, optional, default None 
        maximum memory used during the execcution
    result : None or float, optional, default None 
        output of the function
    error : None or string, optional, default None 
        error of the function
    '''
    text_area.insert(tk.END, f"Function: {function_name}\n")
    if error:
        text_area.insert(tk.END, f"Error: {error}\n\n", 'error')
    else:
        if elapsed_time is not None:
            text_area.insert(tk.END, f"Execution time: {elapsed_time:.6f} seconds\n")
        if current_memory is not None and peak_memory is not None:
            text_area.insert(tk.END, f"Memory usage: Current={current_memory / 1e6:.6f} MB; Peak={peak_memory / 1e6:.6f} MB\n")
        if result is not None:
            text_area.insert(tk.END, f"Result: {result}\n\n")
        else :
            text_area.insert(tk.END, "\n")
    
    text_area.see(tk.END) # For scrolling


def run_tests():
    file_to_test = file_entry.get()                                        # Read file name
    TestEsercizio.file_to_test = file_to_test                              # Set name of the file to analyze
    buffer = io.StringIO()                                                 # Buffer, where we write the information, instead of console
    runner = unittest.TextTestRunner(stream=buffer, verbosity=2)           # Create a runner that write on the buffer 
    suite  = unittest.TestLoader().loadTestsFromTestCase(TestEsercizio)    # Load the test
    runner.run(suite)                                                      # Execution
    output = buffer.getvalue()                                             # Read output of test from buffer
    text_area.insert(tk.END, output)                                       # Write the output on the GUI
    text_area.see(tk.END)                                                  # Scroll down
    buffer.close()                                                         # Close buffer


# Creation of the GUI
root = tk.Tk()
root.title("Test Esercizio")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# Bar for file name
file_label = tk.Label(frame, text="Nome del file da testare:")
file_label.pack(side=tk.LEFT)

file_entry = tk.Entry(frame, width=50)
file_entry.pack(side=tk.LEFT, padx=5)

# Button to execute the function tun_test
run_button = tk.Button(frame, text="Run Tests", command=run_tests)
run_button.pack(side=tk.LEFT, padx=5)

# Area with scrolling
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=160, height=45)
text_area.pack(padx=10, pady=10)

# Style for error, red so is easy to identify
text_area.tag_config('error', foreground='red')

root.mainloop()