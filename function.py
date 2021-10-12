import numpy as np
import matplotlib.pyplot as plt
import math

def function_maps(func_name, **kwargs):
    if func_name == "logistic":
        return kwargs["r"] * kwargs["x"] * (1 - kwargs["x"])
    elif func_name == "tent":
        tmp = np.where(kwargs["x"] < 0.5, kwargs["x"], 1 - kwargs["x"])
        return kwargs["r"]*tmp
    elif func_name == "cubic":
        return kwargs["r"]*(kwargs["x"]**2)*(1-kwargs["x"])
    elif func_name == "sine":
        return kwargs["r"]*np.sin(math.pi*kwargs["x"])
    elif func_name == "exponential":
        return kwargs["x"]*np.exp(kwargs["r"]*(1-kwargs["x"]))
    elif func_name == "quotient":
        return kwargs["r"]*kwargs["x"]/((1-kwargs["x"])**kwargs["beta"])
    else:
        return "This function is not defined"


def plot_values(r, x0, n, f, beta=None, ax=None):
    X = []

    x = x0
    for i in range(n):
        y = function_maps(func_name=f, r=r, x=x, beta=beta)
        X.append(x)
        x = y
    ax.plot(X, '-o')
    ax.set_xlim(0, n)
    ax.set_title('r = {}'.format(r) + ', x_0= {}'.format(x0))
    ax.set_xlabel('n= {}'.format(n))
    ax.set_ylabel('x')



fig1 , (( ax1 , ax2 , ax3 ) , ( ax4 , ax5 , ax6 ) ) = plt . subplots (2
, 3 , figsize =( 12 , 6 ) )
plot_values (1 , 0.5 , 100 , 'logistic', ax= ax1 )
plot_values ( 2.5 , 0.5 , 100 , 'logistic', ax= ax2 )
plot_values ( 3.2 , 0.4 , 100 , 'logistic', ax= ax3 )
plot_values ( 3.5 , 0.4 , 100 , 'logistic', ax= ax4 )
plot_values ( 3.8 , 0.4 , 100 , 'logistic', ax= ax5 )
plot_values(5 , 0.4 , 100 , 'logistic', ax= ax6 )
fig1.suptitle('Logistic map')
plt.show ()

fig2 , (( ax1 , ax2 , ax3 ) , ( ax4 , ax5 , ax6 ) ) = plt . subplots (2
, 3 , figsize =( 12 , 6 ) )
plot_values (1 , 0.5 , 100 , 'tent', ax= ax1 )
plot_values ( 2.5 , 0.5 , 100 , 'tent', ax= ax2 )
plot_values ( 3.2 , 0.4 , 100 , 'tent', ax= ax3 )
plot_values ( 3.5 , 0.4 , 100 , 'tent', ax= ax4 )
plot_values ( 3.8 , 0.4 , 100 , 'tent', ax= ax5 )
plot_values(5 , 0.4 , 100 , 'tent', ax= ax6 )
fig2.suptitle('Tent map')
plt.show ()

fig3 , (( ax1 , ax2 , ax3 ) , ( ax4 , ax5 , ax6 ) ) = plt . subplots (2
, 3 , figsize =( 12 , 6 ) )
plot_values (1 , 0.5 , 100 , 'cubic', ax= ax1 )
plot_values ( 2.5 , 0.5 , 100 , 'cubic', ax= ax2 )
plot_values ( 3.2 , 0.4 , 100 , 'cubic', ax= ax3 )
plot_values ( 3.5 , 0.4 , 100 , 'cubic', ax= ax4 )
plot_values ( 3.8 , 0.4 , 100 , 'cubic', ax= ax5 )
plot_values(5 , 0.4 , 100 , 'cubic', ax= ax6 )
fig3.suptitle('Cubic map')
plt.show ()


fig4 , (( ax1 , ax2 , ax3 ) , ( ax4 , ax5 , ax6 ) ) = plt . subplots (2
, 3 , figsize =( 12 , 6 ) )
plot_values (1 , 0.5 , 100 , 'sine', ax= ax1 )
plot_values ( 2.5 , 0.5 , 100 , 'sine', ax= ax2 )
plot_values ( 3.2 , 0.4 , 100 , 'sine', ax= ax3 )
plot_values ( 3.5 , 0.4 , 100 , 'sine', ax= ax4 )
plot_values ( 3.8 , 0.4 , 100 , 'sine', ax= ax5 )
plot_values(5 , 0.4 , 100 , 'sine', ax= ax6 )
fig4.suptitle('Sine map')
plt.show ()

fig5 , (( ax1 , ax2 , ax3 ) , ( ax4 , ax5 , ax6 ) ) = plt . subplots (2
, 3 , figsize =( 12 , 6 ) )
plot_values (1 , 0.5 , 100 , 'exponential', ax= ax1 )
plot_values ( 2.5 , 0.5 , 100 , 'exponential', ax= ax2 )
plot_values ( 3.2 , 0.4 , 100 , 'exponential', ax= ax3 )
plot_values ( 3.5 , 0.4 , 100 , 'exponential', ax= ax4 )
plot_values ( 3.8 , 0.4 , 100 , 'exponential', ax= ax5 )
plot_values(5 , 0.4 , 100 , 'exponential', ax= ax6 )
fig5.suptitle('Exponential map')
plt.show ()

fig6, (( ax1 , ax2 , ax3 ) , ( ax4 , ax5 , ax6 ) ) = plt . subplots (2
, 3 , figsize =( 12 , 6 ) )
plot_values (1 , 0.5 , 100 , 'quotient', beta=5, ax= ax1 )
plot_values ( 2.5 , 0.5 , 100 , 'quotient', beta=5, ax= ax2 )
plot_values ( 3.2 , 0.4 , 100 , 'quotient', beta=5, ax= ax3 )
plot_values ( 3.5 , 0.4 , 100 , 'quotient', beta=5, ax= ax4 )
plot_values ( 3.8 , 0.4 , 100 , 'quotient', beta=5, ax= ax5 )
plot_values(5 , 0.4 , 100 , 'quotient', beta=5, ax= ax6 )
fig6.suptitle('Quotient map')
plt.show ()

def draw_system(r, x0, n, f, beta=None, ax=None):
    t = np.linspace(0,1)
    y = function_maps(func_name=f, r=r, x=t, beta=beta)
    ax.plot(t, y, 'k', lw=2)
    ax.plot([0,1], [0,1], 'k', lw=2)

    x=x0

    for i in range(n):
        y = function_maps(func_name=f, r=r, x=x, beta=beta)
        # Plot the two lines.
        ax.plot([x, x], [x, y], 'k', lw=1)
        ax.plot([x, y], [y, y], 'k', lw=1)
        # Plot the positions with increasing
        # opacity.
        ax.plot([x], [y], 'ok', ms=10,
                alpha=(i + 1) / n)
        x = y
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_title(f"$r={r:.1f}, \, x_0={x0:.1f}$")



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
draw_system(3.5, .1, 10, 'logistic', ax=ax2)
draw_system(2.5, .1, 10, 'logistic', ax=ax1)
fig.suptitle('Logistic map')
plt.show()


fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
draw_system(0.5, .1, 10, 'tent', ax=ax2)
draw_system(1.5, .1, 10, 'tent', ax=ax1)
fig1.suptitle('Tent map')
plt.show()

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
draw_system(0.5, .1, 10, 'sine', ax=ax2)
draw_system(0.9, .1, 10, 'sine', ax=ax1)
fig2.suptitle('Sine map')
plt.show()

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
draw_system(0.5, .1, 10, 'exponential', ax=ax2)
draw_system(0.3, .1, 10, 'exponential', ax=ax1)
fig3.suptitle('Exponential map')
plt.show()

fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
draw_system(0.6, .1, 10, 'quotient', beta=5,  ax=ax2)
draw_system(0.4, .1, 10, 'quotient', beta=5, ax=ax1)
fig4.suptitle('Quotient map')
plt.show()

fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
draw_system(6, .1, 10, 'cubic', ax=ax2)
draw_system(5, .1, 10, 'cubic', ax=ax1)
fig5.suptitle('Cubic map')
plt.show()

def compute_lyapunov(x, f, r, n, iterations, last , beta=None):
    lyapunov = np.zeros(n)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))
    for i in range(iterations):
        x = function_maps(func_name=f, r=r, x=x, beta=beta)
        # We compute the partial sum of the
        # Lyapunov exponent.
        lyapunov += np.log(abs(r - 2 * r * x))
        # We display the bifurcation diagram.
        if i >= (iterations - last):
            ax1.plot(r, x, ',k', alpha=.25)
    ax1.set_title("Bifurcation diagram")

    # We display the Lyapunov exponent.
    # Horizontal line.
    ax2.axhline(0, color='k', lw=.5, alpha=.5)
    # Negative Lyapunov exponent.
    ax2.plot(r[lyapunov < 0],
             lyapunov[lyapunov < 0] / iterations,
             '.k', alpha=.5, ms=.5)
    # Positive Lyapunov exponent.
    ax2.plot(r[lyapunov >= 0],
             lyapunov[lyapunov >= 0] / iterations,
             '.r', alpha=.5, ms=.5)

    ax2.set_title("Lyapunov exponent")
    plt.tight_layout()
    plt.show()

compute_lyapunov(0.5, "logistic", np.linspace(2.5 , 5.0, 1000), 1000, 1000, 100)
compute_lyapunov(0.5, "tent", np.linspace(1, 2.0, 1000), 1000, 1000, 100)
compute_lyapunov(0.5, "sine", np.linspace(2.5 , 5.0, 1000), 1000, 1000, 100)
compute_lyapunov(0.5, "exponential", np.linspace(2.5 , 5.0, 1000), 1000, 1000, 100)
for _ in np.linspace(5, 100, 20):
    print(_)
    compute_lyapunov(0.4, "quotient", np.linspace(0.1, 100, 10000), 10000, 10000, 100, _)
compute_lyapunov(0.5, "cubic", np.linspace(1 , 10, 10000), 10000, 10000, 100)
