import numpy as np
import matplotlib.pyplot as plt


class PMT:
    def __init__(self, dimension, size, source_x, source_y, source_z):
        self.dimension = dimension
        self.size = size
        self.source = np.array([source_x, source_y, source_z])

    @property
    def get_centers(self):
        sectors = np.empty((0, 3))
        sector_id = -1

        n = self.dimension

        if (n % 2) == 0:  # for even numbers of rows and columns
            for i in range(self.dimension):
                y_center = self.dimension / 2 * self.size - self.size / 2 - i * self.size
                for j in range(self.dimension):
                    sector_id += 1
                    x_center = -1 * self.size * self.dimension / 2 + self.size / 2 + j * self.size
                    sectors = np.append(sectors, np.array([[sector_id, x_center, y_center]]), axis=0)

        else:  # for odd number of rows and columns
            for i in range(self.dimension):
                y_center = (self.dimension - 1) / 2 * self.size - i * self.size
                for j in range(self.dimension):
                    sector_id += 1
                    x_center = (1 - self.dimension) / 2 * self.size + j * self.size
                    sectors = np.append(sectors, np.array([[sector_id, x_center, y_center]]), axis=0)
        return sectors

    @property
    def position_identifier(self):
        identifiers = np.empty((0, 4))
        for item in self.get_centers:
            x_in = False
            if self.source[0] < item[1] - 0.5 * self.size:
                capital_a = item[1] - 0.5 * self.size - self.source[0]
                # print("outside x value of pixel ", int(item[0]), "; A =", capital_a)
            elif self.source[0] > (item[1] + 0.5 * self.size):
                capital_a = self.source[0] + item[1] + 0.5 * self.size
                # print("outside x value of pixel ", int(item[0]), "; A =", capital_a)
            else:
                x_in = True
                capital_a = 0.5 * self.size - abs(self.source[0] - item[1])
                # print("inside x value of pixel ", int(item[0]), "; A =", capital_a)

            y_in = False
            if self.source[1] < item[2] - 0.5 * self.size:
                capital_b = item[2] - 0.5 * self.size - self.source[1]
                # print("outside y value of pixel ", int(item[0]), "; B =", capital_b)
            elif self.source[1] > (item[2] + 0.5 * self.size):
                capital_b = self.source[1] - item[2] - 0.5 * self.size
                # print("outside y value of pixel ", int(item[0]), "; B =", capital_b)
            else:
                y_in = True
                capital_b = 0.5 * self.size - abs(self.source[1] - item[2])
                # print("inside y value of pixel ", int(item[0]), "; B =", capital_b)

            identifiers = np.append(identifiers, np.array([[x_in, y_in, capital_a, capital_b]]), axis=0)

        # print(identifiers)
        return identifiers

    @property
    def get_d_omega(self):

        def omega_main(a, b, d):

            def alpha(aa, dd):
                return aa / (2 * dd)

            def beta(bb, dd):
                return bb / (2 * dd)

            return 4 * np.arctan(alpha(a, d) * beta(b, d)
                                 / np.sqrt(1 + alpha(a, d) ** 2 + beta(b, d) ** 2))

        d_omegas = np.empty(0)

        for item in self.position_identifier:
            capital_a = item[2]
            capital_b = item[3]
            if item[0]:
                if item[1]:  # inside x area inside y area
                    d_omega = (omega_main(2 * (self.size - capital_a), 2 * (self.size - capital_b), self.source[2])
                               + omega_main(2 * capital_a, 2 * (self.size - capital_b), self.source[2])
                               + omega_main(2 * (self.size - capital_a), 2 * capital_b, self.source[2])
                               + omega_main(2 * capital_a, 2 * capital_b, self.source[2])) / 4
                else:  # inside x area, outside y area
                    d_omega = (omega_main(2 * (self.size - capital_a), 2 * (capital_b + self.size), self.source[2])
                               + omega_main(2 * capital_a, 2 * (capital_b + self.size), self.source[2])
                               - omega_main(2 * (self.size - capital_a), 2 * capital_b, self.source[2])
                               - omega_main(2 * capital_a, 2 * capital_b, self.source[2])) / 4
            else:
                if item[1]:
                    d_omega = (omega_main(2 * (capital_a + self.size), 2 * (self.size - capital_b), self.source[2])
                               - omega_main(2 * capital_a, 2 * (self.size - capital_b), self.source[2])
                               + omega_main(2 * (capital_a + self.size), 2 * capital_b, self.source[2])
                               - omega_main(2 * capital_a, 2 * capital_b, self.source[2])) / 4
                else:
                    d_omega = (omega_main(2 * (capital_a + self.size), 2 * (capital_b + self.size), self.source[2])
                               - omega_main(2 * capital_a, 2 * (capital_b + self.size), self.source[2])
                               - omega_main(2 * (capital_a + self.size), 2 * capital_b, self.source[2])
                               + omega_main(2 * capital_a, 2 * capital_b, self.source[2])) / 4

            d_omegas = np.append(d_omegas, np.array([d_omega / (4 * np.pi)]), axis=0)
        d_omegas = np.reshape(d_omegas, (self.dimension, self.dimension))
        return d_omegas


# Fill in: (dimension (nxn grid fill in 'n'), size (fill in sector size), source_x, source_y, source_z)
PMT1 = PMT(2, 24.25, -12.125, 12.125, 12.125)

data = PMT1.get_d_omega
print(data)


def visualization(d):
    fig = plt.figure(figsize=(6, 6))

    ax = fig.add_subplot(111)
    ax.set_title(r'$d \ \ \Omega$')

    plt.imshow(d, interpolation='none')
    ax.set_aspect('equal')

    for (i, j), z in np.ndenumerate(d):
        ax.text(j, i, '{:0.4f}'.format(z), ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.8'))

    plt.axis('off')

    plt.colorbar(orientation='vertical')
    plt.show()


visualization(data)
