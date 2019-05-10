import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt, mpld3

PORT = 8081

def create_test_plot():
    global plot_as_html
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot([1, 2, 3, 4])
    ax1.set_xlabel('Testx')
    ax1.set_ylabel('Testy')
    ax1.set_title('Test title')
    ax1.legend()
    return mpld3.fig_to_html(fig)

plot= create_test_plot()
plot_as_html=mpld3.fig_to_html(plot)

TEMPLATE_FILE = "index.html"
templateLoader = FileSystemLoader( searchpath="." )
templateEnv = Environment( loader=templateLoader )
template = templateEnv.get_template(TEMPLATE_FILE)
html_output = template.render(title="Some title with the template engine.", plot=plot_as_html)

class myHandler(BaseHTTPRequestHandler):
    #Handler for the GET requests
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
        # Send the html message
        self.wfile.write(bytes(html_output, "utf-8"))
        return

httpd = socketserver.TCPServer(("", PORT), myHandler)
print("Server is up and running!")
httpd.serve_forever()
