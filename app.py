from flask import Flask, render_template, request, jsonify, render_template_string
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
import io

omega = 7.292e-5
lat = np.pi/2
f = 2*omega*np.sin(lat)

app = Flask(__name__, static_url_path="")

if __name__ == Flask(__name__):
	app.debug = True

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/simulasi")
def simulasi():
    return render_template("simulasi.html")

@app.route("/quiz")
def quiz():
    return render_template("quiz-section.html")

@app.route("/teori1")
def teori1():
    return render_template("blog-single.html")

@app.route("/teori2")
def teori2():
    return render_template("blog-single2.html")

@app.route("/teori3")
def teori3():
    return render_template("blog-single3.html")

@app.route("/tim1")
def tim1():
    return render_template("team1.html")

@app.route("/tim2")
def tim2():
    return render_template("team2.html")

@app.route("/tim3")
def tim3():
    return render_template("team3.html")

@app.route("/information")
def information():
    return render_template("information.html")

def stream(x, y, U, v, num, hline):
    xsize = x.shape[0]
    ysize = y.shape[0]
    miny = y[0,0]
    maxy = y[-1,0]
    minx = x[0,0]
    maxx = x[0,-1]

    dx = (maxx-minx)/xsize
    dy = (maxy-miny)/ysize
    dh = ysize/num

    tstep = dx/U
    mtncolor = [0.02, 0.77, 0.02]

    fig = plt.figure()
    ycell = 1
    for j in range(0, num):
        ycell = 1+dh*(j)
        if ycell < 0:
            ycell = 0
        
        if ycell > ysize:
            ycell = ysize

        ax = []
        ay = []
        ax.append(minx)
        ycell = int(round(ycell))
        ycell = ycell - 1
        ay.append(y[ycell,0])

        for i in range(1, xsize):
            ax.append(x[ycell,i])
            ay.append(ay[i-1]+tstep*v[ycell,i])

        if j == 0:
            ax.append(maxx)
            ay.append(miny)
            plt.fill_between(ax, ay, 0, color=mtncolor)
        else:
            plt.plot(ax, ay, "b-")

        plt.grid(True)
        plt.ylim(miny, maxy)
        plt.xlim(minx, maxx)
        plt.yticks([0.,1000.,2000.,3000.,4000.,5000.,6000.,
            7000.,8000.,9000.,10000.]) #np.arange(0,11000,1000))
        plt.xlabel("X (m)")
        plt.ylabel("Ketinggian (m)")
        plt.title("Analisis Streamline")
        plt.plot(x[0,:], hline, "m--")
        del(ax)
        del(ay)

    img_stream = io.BytesIO()
    fig.savefig(img_stream, format="png")
    img_stream.seek(0)
    img_stream_base64 = base64.b64encode(img_stream.read()).decode("utf8")

    return img_stream_base64

def tlwplot(Lupper, Llower, U, H, a, ho, xdom, zdom, mink, maxk):
    npts = 40
    dk = 0.367/a            # interval bilangan gelombang k
    nk = (maxk-mink)/dk
    minx = -0.25*xdom
    maxx = 0.75*xdom
    minz = 0
    maxz = zdom

    matrix1 = np.zeros((npts+1, npts+1))
    matrix2 = np.zeros((npts+1, npts+1))
    matrix3 = np.zeros((npts+1, npts+1))

    dx = (maxx - minx)/npts
    dz = (maxz - minz)/npts

    x = np.arange(minx, maxx+dx, dx)
    z = np.arange(minz, maxz+dz, dz)
    k = np.arange(mink, maxk, dk)
    x,z = np.meshgrid(x, z)
    ht = 0
    for kloop in range(0, int(nk)):
        kk = k[kloop]
        m = np.lib.scimath.sqrt(Llower*Llower-kk*kk) if (Llower*Llower-kk*kk) < 0 else np.sqrt(Llower*Llower-kk*kk)
        n = np.lib.scimath.sqrt(kk*kk-Lupper*Lupper) if (kk*kk-Lupper*Lupper) < 0 else np.sqrt(kk*kk-Lupper*Lupper)
        if (m+1j*n == 0):
            r = 9e99
        else:
            r = (m-1j*n)/(m+1j*n)

        R = r*np.exp(2*1j*m*H)
        A = (1+r)*np.exp(H*n+1j*H*m)/(1+R)
        C = 1/(1+R)
        D = R*C
        hs = np.pi*a*ho*np.exp(-a*np.abs(kk))
        ht = ht+np.pi*dk*a*np.exp(-a*np.abs(kk))

        aboveH = A*np.exp(-z*n)*(z>H)
        belowH = (C*np.exp(1j*z*m)+D*np.exp(-1j*z*m))*(z<=H)

        matrix2 = ((-1j*kk*hs*U*(aboveH+belowH))*np.exp(-1j*x*kk))
        if kloop > 1:
            matrix3 = matrix3+0.5*(matrix1+matrix2)*dk

        matrix1 = matrix2
    
    w = np.real(matrix3/ht)
    Hline = H*np.ones((npts+1))
    fig_stream = stream(x, z, U, w, 10, Hline)
    #plt.plot(x[0,:], Hline, "m--")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bounds = [-10.,-8.,-6.,-4.,-2.,0.,2.,4.,6.,8.,10.] #np.arange(-10,10+2,2)
    cbounds = np.arange(-10,10.01,0.2)
    cnf = ax.contourf(x, z, w, cbounds, vmin=-10, vmax=10, interpolation=None,cmap="jet", extend="both")
    line = ax.plot(x[0,:], Hline, "m--")
    plt.xlabel("X (m)")
    plt.ylabel("Ketinggian (m)")
    plt.title("Kecepatan Vertikal (m/s)")
    plt.yticks([0.,1000.,2000.,3000.,4000.,5000.,6000.,7000.,
        8000.,9000.,10000.,]) #np.arange(0,11000,1000))
    cnf.set_clim(-10,10)
    cb = plt.colorbar(cnf,ticks=bounds,extend="min")

    img_tlw = io.BytesIO()
    fig.savefig(img_tlw, format="png")
    img_tlw.seek(0)
    img_tlw_base64 = base64.b64encode(img_tlw.read()).decode("utf8")

    return img_tlw_base64, fig_stream

@app.route("/update_rossby", methods=["GET", "POST"])
def update_rossby():
    data = request.json
    wspd = float(data.get("wspd_value"))
    hlfwdt = float(data.get("hlfwdt_value"))
    U = wspd
    a = 1000*hlfwdt
    rossby = U/(f*a)
    rossby = "{0:.4f}".format(rossby)
    return str(rossby)

@app.route("/update_scorer", methods=["GET", "POST"])
def update_scorer():
    data = request.json
    intht = float(data.get("intht_value"))
    llower = float(data.get("llower_value"))
    lupper = float(data.get("lupper_value"))
    H = 1000*intht
    Llower = 0.0001*llower
    Lupper = 0.0001*lupper
    scorer = 4*H*H*(Llower*Llower-Lupper*Lupper)/(np.pi*np.pi)
    scorer = "{0:.4f}".format(scorer)
    return str(scorer)

@app.route("/analyzeFlow", methods=["GET", "POST"])
def analyzeFlow():
    data = request.json
    hlfwdt = float(data.get("hlfwdt_value"))
    intht = float(data.get("intht_value"))
    maxhgt = float(data.get("maxhgt_value"))
    wspd = float(data.get("wspd_value"))
    lupper = float(data.get("lupper_value"))
    llower = float(data.get("llower_value"))
    hdom = float(data.get("hdom_value"))
    vdom = float(data.get("vdom_value"))
    smink = float(data.get("smink_value"))
    smaxk = float(data.get("smaxk_value"))

    a = 1000*hlfwdt
    H = 1000*intht
    ho = 1000*maxhgt
    U = wspd
    Lupper = 0.0001*lupper
    Llower = 0.0001*llower
    xdom = 1000*hdom
    zdom = 1000*vdom
    mink = smink/a
    maxk = smaxk/a

    tlw, stream = tlwplot(Lupper, Llower, U, H, a, ho, xdom, zdom, mink, maxk)
        
    return jsonify({"tlw_plot": tlw, "stream_plot": stream})

@app.route("/showResults", methods=["POST"])
def show():
    tlw_data = request.form.get('tlw_data')
    stream_data = request.form.get('stream_data')

    html_template = """
    <!DOCTYPE html>
    <html lang="en">

    <head>
    <meta charset="utf-8">
    <meta content="width=device-width, initial-scale=1.0" name="viewport">

    <title>Maestro</title>
    <meta content="" name="description">
    <meta content="" name="keywords">

    <!-- Favicons -->
    <link href="assets/img/logo maestro.png" rel="icon">
    <link href="assets/img/logo maestro.png" rel="apple-touch-icon">

    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css?family=Open+Sans:300,300i,400,400i,600,600i,700,700i|Jost:300,300i,400,400i,500,500i,600,600i,700,700i|Poppins:300,300i,400,400i,500,500i,600,600i,700,700i" rel="stylesheet">

    <!-- Vendor CSS Files -->
    <link href="assets/vendor/aos/aos.css" rel="stylesheet">
    <link href="assets/vendor/bootstrap/css/bootstrap.min.css" rel="stylesheet">
    <link href="assets/vendor/bootstrap-icons/bootstrap-icons.css" rel="stylesheet">
    <link href="assets/vendor/boxicons/css/boxicons.min.css" rel="stylesheet">
    <link href="assets/vendor/glightbox/css/glightbox.min.css" rel="stylesheet">
    <link href="assets/vendor/remixicon/remixicon.css" rel="stylesheet">
    <link href="assets/vendor/swiper/swiper-bundle.min.css" rel="stylesheet">

    <!-- Template Main CSS File -->
    <link href="assets/css/style.css" rel="stylesheet">

    <!-- =======================================================
    * Template Name: Sistem Informasi Meteorologi
    * Updated: Mar 10 2023 with Bootstrap v5.2.3
    * Template URL: https://bootstrapmade.com/arsha-free-bootstrap-html-template-corporate/
    * Author: BootstrapMade.com
    * License: https://bootstrapmade.com/license/
    ======================================================== -->
    </head>

    <body>

    <!-- ======= Header ======= -->
    <header id="header" class="fixed-top header-inner-pages">
        <div class="container d-flex align-items-center">
        <!--<h1 class="logo me-auto"><a href="/">MAESTRO</a></h1> -->
        <!-- Uncomment below if you prefer to use an image logo -->
        <a href="/" class="logo me-auto"><img src="assets/img/logo maestro.png" alt="" class="img-fluid"></a>
        <nav id="navbar" class="navbar">
            <ul>
            <li><a class="nav-link scrollto" href="../#hero">Beranda</a></li>
            <li><a class="nav-link scrollto" href="../#about">Tentang Maestro</a></li>
            <li><a class="nav-link scrollto" href="../#kepointeori">Tentang Gelombang Gunung</a></li>
            <li><a class="nav-link scrollto" href="/quiz">Kuis</a></li>
            <li><a class="nav-link scrollto" href="/simulasi">Simulasi</a></li>
            <li><a class="nav-link scrollto" href="../#team">Tim</a></li>
            <li><a class="nav-link scrollto" href="../#contact">Kontak</a></li>
            <li><a class="getstarted scrollto" href="../#about">Mulai</a></li>
            </ul>
            <i class="bi bi-list mobile-nav-toggle"></i>
        </nav><!-- .navbar -->
        </div>
    </header><!-- End Header -->

    <main id="main">

    <!-- Team Section -->
    <section id="analisis" class="team section">

    <!-- Section Title -->
    <div class="container section-title" data-aos="fade-up">
        <br/><br/><br/>
        <h2>Hasil Analisis</h2>
    </div><!-- End Section Title -->

    <div class="container">
        <div class="row content">
        <div class="col-lg-6">
            <div class="flex-container">
                <img id="tlw" src="data:image/png;base64,{{ tlw_data }}">
            </div>
        </div><!-- End Team Member -->

        <div class="col-lg-6">
            <div class="flex-container">
                <img id="stream" src="data:image/png;base64,{{ stream_data }}">
            </div>
        </div><!-- End Team Member -->
        </div>
    </div>

    </section>

    </main>

    <div id="preloader"></div>
    <a href="#" class="back-to-top d-flex align-items-center justify-content-center"><i class="bi bi-arrow-up-short"></i></a>

    <!-- Vendor JS Files -->
    <script src="assets/vendor/aos/aos.js"></script>
    <script src="assets/vendor/bootstrap/js/bootstrap.bundle.min.js"></script>
    <script src="assets/vendor/glightbox/js/glightbox.min.js"></script>
    <script src="assets/vendor/isotope-layout/isotope.pkgd.min.js"></script>
    <script src="assets/vendor/swiper/swiper-bundle.min.js"></script>
    <script src="assets/vendor/waypoints/noframework.waypoints.js"></script>
    <script src="assets/vendor/php-email-form/validate.js"></script>

    <!-- Template Main JS File -->
    <script src="assets/js/main.js"></script>

    </body>
    </html>
    """

    return render_template_string(html_template, tlw_data=tlw_data, stream_data=stream_data)

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(debug=True)
