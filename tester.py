# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 20:45:54 2021

@author: Madhav
"""
class Madhav:
    
    def _init_(self, name):
        
        self.name = name
        
    def print_name(self):
        print(self.name)
        
if __name__ == "__main__":
    
    person = Madhav("Madhav")
    person.print_name()