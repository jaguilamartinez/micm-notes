# -*- coding: utf-8 -*-

def plot_decision_surface(clf, features, labels, title):
    x_min = -4.0; x_max = 12.0
    y_min = -10.0; y_max = 4.0

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])   
    z_min, z_max = -np.abs(Z).max(), np.abs(Z).max()
    
    # set title
    f, ax = plt.subplots()
    ax.set_title(title)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.pcolormesh(xx, yy, Z, cmap='gray', vmin=.2, vmax=2.4)  # 

    # Plot the points
    class_1_x = [features[ii][0] for ii in range(0, len(features)) if labels[ii]==1]
    class_1_y = [features[ii][1] for ii in range(0, len(features)) if labels[ii]==1]
    class_2_x = [features[ii][0] for ii in range(0, len(features)) if labels[ii]==2]
    class_2_y = [features[ii][1] for ii in range(0, len(features)) if labels[ii]==2]

    plt.scatter(class_1_x, class_1_y, color = '#F5CA0C', marker='v', label="1", s=40)
    plt.scatter(class_2_x, class_2_y, color = '#00A99D', marker='x', label="2", s=40)
   
    plt.xlabel("feature_a")
    plt.ylabel("feature_b")
    plt.legend(('$1st Ds - A$', '$2nd Ds - B$'), frameon=True)