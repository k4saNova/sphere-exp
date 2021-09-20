import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat


def get_boundary(clf):
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = 0., 1.
    y_min, y_max = 0., 1.
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return xx, yy, Z


def plot_boundary(X, y, clf, labels=["x1", "x2"], plot_data=True,
                  show=True, save=False, file_name="out.png"):
    if X.shape[1] != 2:
        return None
    ax = plt.axes()
    xx, yy, Z = get_boundary(clf)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
    
    # Put the result into a color plot
    plt.figure(1, figsize=(4, 4))
    # Plot also the training points
    if plot_data:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    ax.set_aspect("equal")
    if save:
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.clf()


def plot_circles(X, y, spheres, clf, labels=["x1", "x2"], plot_data=False,
                 show=True, save=False, file_name="out.png"):
    if X.shape[1] != 2:
        return None
    
    fig = plt.figure()
    ax = plt.axes()
    xx, yy, Z = get_boundary(clf)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
    if plot_data:
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
        
    for s in spheres.values():
        # print(s)
        c = pat.Circle(xy=s[0], radius=s[1], fill=False)
        ax.add_patch(c)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_aspect("equal")
    
    if save:
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.clf()
