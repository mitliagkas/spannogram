from mrjob.job import MRJob
import mrjob.protocol

# A protocol capable of reading the flickr database, tentatively
# named 'the flickr protocol'
class FlickrProtocol(object):
    def read(self, line):
        if line[0]=='#':
            return None, None
        else:
            e1_str, e2_str = line.split(' ', 1)
            return [int(e1_str), int(e2_str)], 1

    def write(self, key, value):
        return '%s\t%s' % key, value

class MRPowerMethod(MRJob):
    INTERNAL_PROTOCOL = mrjob.protocol.JSONProtocol
    OUTPUT_PROTOCOL = mrjob.protocol.JSONProtocol

    def configure_options(self):
        super(MRPowerMethod, self).configure_options()
        self.add_passthrough_option('--components', default=2,
            help="Number of principal components to extract.")
        self.add_passthrough_option('--iterations', default=2,
            help="Number of iterations of the power method to do.")
        self.add_passthrough_option('--partitions', default=10,
            help="Number of partitions for the reducers."
                  " A good guideline is 2 per core.")
        self.add_passthrough_option('--diagonal', default=0,
            help="Amount to boost the diagonal by.")
        self.add_passthrough_option('--input-protocol',
            default='JSON', choices=['JSON', 'flickr'],
            help="Protocol to use for input.")
        self.add_passthrough_option('--symmetricize', default=True,
            help="If true, every input edge will also yield its "
                 "symmetric in the first mapper's output.")

    def load_options(self, args):
        super(MRPowerMethod, self).load_options(args=args)
        if self.options.input_protocol=='JSON':
            self.INPUT_PROTOCOL=mrjob.protocol.JSONProtocol
        elif self.options.input_protocol=='flickr':
            self.INPUT_PROTOCOL=FlickrProtocol
        else:
            self.INPUT_PROTOCOL=mrjob.protocol.JSONProtocol

    def mapper(self, coordinates, value, initRun=False):
        numberOfPartitions=int(self.options.partitions)
        if coordinates is None:
            return
        [i,j]=coordinates
        if j>0: # An element of the matrix A - use the row number
                # to hash into right partition
            p=(i % numberOfPartitions)+1
            yield p, [i, j, value]
            if not initRun or self.options.input_protocol=='flickr':
                p=(j % numberOfPartitions)+1
                yield p, [j, i, value]
        else: # Element of a previously established eigvec, the (-j)-th
              # Send to all partitions
            for p in xrange(1,numberOfPartitions+1):
                yield p, [i, j, value]

    def reducer(self, partition, values, finalRun=False):
        numberOfPartitions=int(self.options.partitions)
        numberOfComponents=int(self.options.components)
        A={}
        Vtr={}
        normsqV_k=[0]*numberOfComponents
        for value in values:
            [i,j,v]=value
            if j>0: # A column of the matrix A
                try:
                    A[i][j]=v
                except KeyError:
                    A[i] = {}
                    A[i][j]=v
                if not finalRun:
                    yield partition, value
            else: # Element of a previously established eigvec, the (-j)-th
                try:
                    Vtr[-j][i]=v
                except KeyError:
                    Vtr[-j]={}
                    Vtr[-j][i]=v
                normsqV_k[-j-1] += v**2

        # Now compute this shit for every i that made it to this
        # reducer
        for k in xrange(1,numberOfComponents+1):
            # Only normalize and deflate the k-th component if it exists
            if k in Vtr:
                for j in Vtr[k]:
                    Vtr[k][j]=Vtr[k][j]/normsqV_k[k-1]**(0.5)
                # Deflate
                for l in xrange(1,k):
                    index=set(Vtr[k].keys()+Vtr[l].keys())
                    innerProduct=sum([ Vtr[k].get(j,0.0)*Vtr[l].get(j,0.0)
                                        for j in index ])
                    for j in Vtr[k].keys():
                        Vtr[k][j]-=innerProduct*Vtr[l].get(j,0.0)
            else:
                Vtr[k]={}

            # Finally compute new V_k's
            for i in A.iterkeys():
                # Add the diagonal element if needed
                dg=float(self.options.diagonal)
                if k==1 and dg>0:
                    #A[i][i]=A[i].get(i,0.0)+dg
                    A[i][i]=dg
                AVik=0
                for j in A[i].iterkeys():
                    try:
                        AVik += float(A[i][j])*Vtr[k][j]
                    except:
                        Vtr[k][j]=self.basis(k,j)
                        AVik += float(A[i][j])*Vtr[k][j]
                if not finalRun:
                    for p in xrange(1,numberOfPartitions+1):
                        yield p, [i, -k, AVik]
                else:
                    yield [i,k], AVik

    # This mapper-reducer pair is used in a last, cheap, step to
    # provide results in the form: index, [v_1(index), v_2(index),...]
    def merge_mapper(self, coordinates, value):
        if coordinates is None:
            return
        [i,k]=coordinates
        yield i, [k, value]

    def merge_reducer(self, coordinate, values):
        numberOfComponents=int(self.options.components)
        outValues=[0]*numberOfComponents
        for value in values:
            [k,v]=value
            outValues[k-1]=v
        yield coordinate, outValues

    # Returns elements from a predefined orthogonal (-ish) basis (not
    # normalized)
    def basis(self,k,j):
        if k==1:
            return 1
        else:
            ind=(j-1)/2**(k-2)
            return (-1)**(ind+1)

    def steps(self):
        totalIterations=int(self.options.iterations)
        def finalReducer(key,values):
            return self.reducer(key,values,True)
        def initMapper(key,values):
            return self.mapper(key,values,True)

        return [self.mr(mapper=initMapper, reducer=self.reducer)]\
                + [
                    self.mr(reducer=self.reducer)
                  ]*(totalIterations-2)\
                + [self.mr(reducer=finalReducer)]\
                + [self.mr(mapper=self.merge_mapper,
                           reducer=self.merge_reducer)]

if __name__ == '__main__':
    MRPowerMethod.run()
