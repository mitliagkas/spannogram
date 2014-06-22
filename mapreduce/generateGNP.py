from numpy import random

from mrjob.job import MRJob
#import random

class MRGnpGenerator(MRJob):
    # the following defines the input options used in the MR routine
    def configure_options(self):
        super(MRGnpGenerator, self).configure_options()
        self.add_passthrough_option('--cliqueSize', default=0, help="The size of the planted clique (k)")

    """
    Accepts as input the sequence 1 ... n
    Generates all tuples (i,j),Bin(1/2), for j<i
    """
    def mapper(self, _, value):
        k=int(self.options.cliqueSize)
        i=int(value)
        for j in xrange(1,int(value)):
            if i<=k and j<=k and i!=j:
                edge=1
            else:
                edge=int(random.random_integers(0,1,1))
                #edge=random.randint(0,1)
            if edge==1:
                yield [i,j],edge
                #yield [j,i],edge


if __name__ == '__main__':
    MRGnpGenerator.run()
