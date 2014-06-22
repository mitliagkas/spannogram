from optparse import OptionError
import math
import heapq
import operator

from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol


class MRSpannogram(MRJob):

    INPUT_PROTOCOL = JSONProtocol
    
    # the following defines the input options used in the MR routine
    def configure_options(self):
        super(MRSpannogram, self).configure_options()
        self.add_passthrough_option('--rowcount', help="Row count of V")
        self.add_passthrough_option('--cliquesize', help="Size of presumed clique.")
    
    # loads and checks options
    def load_options(self, args): 
        super(MRSpannogram, self).load_options(args=args)
        if not self.options.rowcount:
            raise OptionError("All Options must be present")

    def spannogram_mapper(self, coordinate, values):
        i = coordinate
        values.append(i-1)
        for j in range(int(self.options.rowcount)):
            yield j, values


    def spannogram_reducer(self, coordinate, values):
        d = 2
        k = int(self.options.cliquesize)
        Vunsorted = list(values)
        V = [[0]*d]*int(self.options.rowcount)
        i = coordinate

        for l in range(int(self.options.rowcount)):
            V[Vunsorted[l][2]] = Vunsorted[l][0:d]
        
        t1 = 0
        t2 = 0
        for l in range(int(self.options.rowcount)):
            t1 +=(V[l][0]**2)
            t2 +=(V[l][1]**2)
    
        t1 = math.sqrt(math.sqrt(t1))
        t2 = math.sqrt(math.sqrt(t2))
        for l in range(int(self.options.rowcount)):
            V[l][0] = V[l][0]/t1
            V[l][1] = V[l][1]/t2
    
        
        Vc = []
        opt_support = [];
        opt_metric = 0;

        for j in range(i+1, int(self.options.rowcount)):
            x = []
            Vc = []
            Vtemp = []
            # compute c_ij intersection vector
            x = [V[i][l]-V[j][l] for l in range(2)]
            # cumpute v_ij = Vc_ij
            Vc = [V[l][0]*x[1]-V[l][1]*x[0] for l in range(int(self.options.rowcount))]
            # find top and bottom support
            top_support_pos = zip(*heapq.nlargest(k, enumerate(Vc), key=operator.itemgetter(1)))[0]
            top_support_neg = zip(*heapq.nsmallest(k, enumerate(Vc), key=operator.itemgetter(1)))[0]
            # compute top/bottom support metric
            Vtemp = [V[s] for s in top_support_pos]
            Vtemp_sum = [sum(a) for a in zip(*Vtemp)]
            metric_pos = sum([a**2 for a in Vtemp_sum])
            
            Vtemp = [V[s] for s in top_support_neg]
            Vtemp_sum = [sum(a) for a in zip(*Vtemp)]
            metric_neg = sum([a**2 for a in Vtemp_sum])
            # find locally optimal support
            metric_list = [opt_metric, metric_pos, metric_neg]
            metric_index = metric_list.index(max(metric_list))
            opt_support = [opt_support, top_support_pos, top_support_neg][metric_index]
            opt_metric = max(metric_list)

        yield i, [opt_metric, opt_support]

   

    # sequence of map-reduce-reduce...-reduce tasks
    def steps(self):
        return [self.mr(mapper=self.spannogram_mapper,
                        reducer=self.spannogram_reducer)]

# runs initialization and steps
if __name__ == '__main__':
    MRSpannogram.run()
