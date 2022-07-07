


class myparent(object):
    def __init__(self,x):
        self.x=x 

class myclass(myparent):
    def __init__(self,x,y):
        self.y=y
        super().__init__(x) 

myobj = myclass(3,2)
print(myobj.x, myobj.y)