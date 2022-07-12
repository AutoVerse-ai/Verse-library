import inspect 

def dynamics(a,b,c):
    da = a+1
    db = 0
    dc = c+2
    return [da, db, dc]

if __name__ == "__main__":
    print(inspect.getsource(dynamics))