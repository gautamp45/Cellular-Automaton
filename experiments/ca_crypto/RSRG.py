import secrets

class RandomStateRuleGen:
    def __init__(self):
        self.state = ""
        self.key = 0
        self.gen = 0
        self.seed = b'11011011011100110'
        
    def RandomS(self,length):
        
        for _ in range(length):
            bit = secrets.choice(["0", "1"])
            self.state += bit
            
        return(self.state)
    
    def RandomR(self):
        
        secretGenerator  = secrets.SystemRandom(self.seed)
        
        self.rule = secretGenerator.randint(0,256) 
        
        return(self.rule)
    
    def RandomG(self):
        
        secretGenerator  = secrets.SystemRandom(self.seed)
        
        self.gen = secretGenerator.randint(0,1001) 
        
        return(self.gen)
        
        
# r = RandomStateRuleGen()
# print(r.RandomS(10))
# print(r.RandomR())
# print(r.RandomG())
