from graphviz import Digraph

dot = Digraph(comment='AI-Native Network Control', format='png')
dot.attr(rankdir='TB', fontsize='12')

dot.node('Laptop', 'Laptop\n(Sim & Preproc)', shape='box')
dot.node('DataPrep', 'Dataset Prep\n& Preprocessing', shape='box')
dot.node('TrafficGen', 'Traffic Generator\n(health_sender)', shape='box')
dot.node('Network', 'Network\n(loopback / lab switch)', shape='box')
dot.node('Receiver', 'Receiver + Telemetry\n(health_receiver + agent)', shape='box')
dot.node('Telemetry', 'Telemetry Store\n(CSV / TSDB)', shape='box')
dot.node('Features', 'Feature Engineering\n(telemetry → X)', shape='box')
dot.node('Train', 'AI Model Training\n(offline)', shape='box')
dot.node('Controller', 'AI-Native Controller\n(inference)', shape='box')
dot.node('ControlActions', 'Control Actions\n(rate/prio)', shape='box')
dot.node('Lab', 'Networks Lab\n(PCs + Switch + tc/netem)', shape='box')
dot.node('Impair', 'Impairments\n(tc / iperf)', shape='box')
dot.node('Analysis', 'Analysis & Plots', shape='box')
dot.node('Compare', 'Rule vs AI\nComparison', shape='box')
dot.node('Report', 'Report & Demo', shape='box')

dot.edges([('Laptop','DataPrep'), ('DataPrep','Features'), ('Features','Train'),
           ('Train','Controller'), ('Controller','ControlActions'), ('ControlActions','TrafficGen'),
           ('TrafficGen','Network'), ('Network','Receiver'), ('Receiver','Telemetry'), ('Telemetry','Features'),
           ('Telemetry','Analysis'), ('Analysis','Compare'), ('Compare','Report')])

dot.edge('Network','Lab')
dot.edge('Lab','Impair')
dot.edge('Impair','Network')

dot.render('ai_native_network_diagram', view=False)
print("diagram saved as ai_native_network_diagram.png")
