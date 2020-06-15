import random
import torch


class ListPool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_z = 0
            self.num_n = 0
            self.zs = []
            self.ns = []

    def query(self,zs,ns):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return vectors
        
        return_vectors_z = [[] for k in range(len(zs))]
        return_vectors_n = [[] for k in range(len(ns))]

        if self.num_z == 0:
            for k in range(len(zs)):
                self.zs.append([])

        if self.num_n == 0:
            for k in range(len(ns)):
                self.ns.append([])
        
        '''for z in zs:
            if self.num_z < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.zs.append(z)
                tmp_z=[]
                for temp in z:
                    tmp_z.append(temp.clone().detach())
                return_vectors_z.append(torch.stack(tmp_z))
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp_z = []
                    for temp in self.zs[random_id]:
                        tmp_z.append(temp.clone().detach())
                    self.zs[random_id] = z
                    return_vectors_z.append(torch.stack(tmp_z))
                else:        # by another 50% chance, the buffer will return the current image
                    tmp_z = []
                    for temp in z:
                        tmp_z.append(temp.clone().detach())
                    return_vectors_z.append(torch.stack(tmp_z))
        
        for n in ns:
            if self.num_n < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.ns.append(n)
                tmp_n=[]
                for temp in n:
                    tmp_n.append(temp.clone().detach())
                return_vectors_n.append(torch.stack(tmp_n))
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp_n = []
                    for temp in self.ns[random_id]:
                        tmp_n.append(temp.clone().detach())
                    self.ns[random_id] = n
                    return_vectors_n.append(torch.stack(tmp_n))
                else:        # by another 50% chance, the buffer will return the current image
                    tmp_n = []
                    for temp in n:
                        tmp_n.append(temp.clone().detach())
                    return_vectors_z.append(torch.stack(tmp_n))

        self.num_z = self.num_z + len(zs[0])
        self.num_n = self.num_n + len(ns[0])'''

        for k in range(len(zs[0])):
            if self.num_z < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_z = self.num_z + 1
                for i in range(len(zs)):
                    self.zs[i].append(zs[i][k])
                    return_vectors_z[i].append(zs[i][k].clone().detach())
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    #print('self.num_z,len(self.zs[0])',self.num_z,len(self.zs[0]))
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    for i in range(len(zs)):
                        return_vectors_z[i].append(self.zs[i][random_id].clone().detach())
                        self.zs[i][random_id] = zs[i][k]
                else:
                    for i in range(len(zs)):
                        self.zs[i].append(zs[i][k])
                        return_vectors_z[i].append(zs[i][k].clone().detach())
                        

        for k in range(len(ns[0])):
            if self.num_n < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_n = self.num_n + 1
                for i in range(len(ns)):
                    self.ns[i].append(ns[i][k])
                    return_vectors_n[i].append(ns[i][k].clone().detach())
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    for i in range(len(ns)):
                        return_vectors_n[i].append(self.ns[i][random_id].clone().detach())
                        self.ns[i][random_id] = ns[i][k]
                else:
                    for i in range(len(ns)):
                        self.ns[i].append(ns[i][k])
                        return_vectors_n[i].append(ns[i][k].clone().detach())
        for k in range(len(return_vectors_z)):
            return_vectors_z[k] = torch.stack(return_vectors_z[k])
        
        for k in range(len(return_vectors_n)):
            return_vectors_n[k] = torch.stack(return_vectors_n[k])
        #print (len(return_vectors_z),len(return_vectors_n))
        #print('self.num_z,self.num_n)',self.num_z,self.num_n)
        return return_vectors_z,return_vectors_n
