import os

from csv import reader

def read ( dirName , fileName , number , read_before_return = False , skip = True ) :

    rows , i = '' , 1

    basedir = os.path.join ( dirName )

    file_origin = os.path.join ( basedir , fileName + '.csv' )

    with open ( file_origin , 'r' ) as my_file :
        
         file_csv , head = reader ( my_file  ) , None

         if skip:   head = next   ( file_csv )
        
         while i <= number :

               if i > 1 : rows += '\n'

               try :   head = next ( file_csv )
               except: head = None
         
               not_break_bool = ( ( head != None ) and ( i <= number ) )
              
               if     not_break_bool : rows += ','.join ( head )
               if not not_break_bool :                    break
               if     not_break_bool : i += 1

    if read_before_return : print ( rows.split ( '\n' ) [ 0 ] )
    return rows
