from qip.pipeline import run_graph, get_deps
from qip.pipeline import GraphAccumulator
from qip.operators import SwapOp, COp, NotOp


class SvgFeeder(GraphAccumulator):
    BLACKLIST = ["SplitQubit", "Q"]

    def __init__(self, n, linespacing=20, opheight=10, opbuffer=10, linewidth=2, opoutlinewidth=1, fontwidth=7):
        self.svg_acc = ''
        self.n = n
        self.linespacing = linespacing
        self.opheight = opheight
        self.opbuffer = opbuffer
        self.linewidth = linewidth
        self.opoutlinewidth = opoutlinewidth
        self.font_size = fontwidth

        self.last_x_center = 0
        self.last_x_max = 0

        # Ops to draw once lines have been drawn.
        self.ops_to_draw = []

    def trim_op_name(self, node):
        nodestr = repr(node)
        paren_pos = nodestr.find('(')
        if paren_pos > 0:
            nodestr = nodestr[:paren_pos]
        return nodestr

    def get_op_string(self, qbitindex, node, x):
        s = ""
        # Add ops to list with relevant positions.
        node_indices = qbitindex[node]
        if len(node_indices) > 1:
            s += '<line x1="{}" x2="{}" y1="{}" y2="{}" style="stroke:black;stroke-width:{}"></line>\n'.format(
                x, x,
                min(node_indices) * self.linespacing + self.opheight,
                max(node_indices) * self.linespacing + self.opheight,
                self.linewidth / 2
            )

        if type(node) == SwapOp:
            # X at each node swap position. line behind them all.
            for node_index in node_indices:
                center_y = self.linespacing*node_index + self.opheight
                s += '<line x1="{}" x2="{}" y1="{}" y2="{}" style="stroke:black;stroke-width:{}"></line>\n'.format(
                    x - self.opheight/2,
                    x + self.opheight/2,
                    center_y - self.opheight/2,
                    center_y + self.opheight/2,
                    self.linewidth
                )
                s += '<line x1="{}" x2="{}" y1="{}" y2="{}" style="stroke:black;stroke-width:{}"></line>\n'.format(
                    x - self.opheight/2,
                    x + self.opheight/2,
                    center_y + self.opheight/2,
                    center_y - self.opheight/2,
                    self.linewidth
                )
        elif type(node) == COp:
            # First index gets the dot, others drawn as normal with a line behind them all.
            s += '<circle cx="{}" cy="{}" r="{}" stroke="black" stroke-width="1" fill="black" />'.format(
                x, node_indices[0] * self.linespacing + self.opheight,
                self.opheight / 2
            )

            s += self.get_op_string({node.op: qbitindex[node][1:]}, node.op, x)
        elif type(node) == NotOp:
            for node_index in node_indices:
                center_y = node_index * self.linespacing + self.opheight
                s += '<circle cx="{}" cy="{}" r="{}" stroke="black" stroke-width="1" fill="white" />'.format(
                    x, center_y, self.opheight / 2
                )
                s += '<line x1="{}" x2="{}" y1="{}" y2="{}" style="stroke:black;stroke-width:{}"></line>\n'.format(
                    x, x,
                    center_y - self.opheight/2,
                    center_y + self.opheight/2,
                    self.linewidth/2
                )
                s += '<line x1="{}" x2="{}" y1="{}" y2="{}" style="stroke:black;stroke-width:{}"></line>\n'.format(
                    x - self.opheight / 2,
                    x + self.opheight / 2,
                    center_y, center_y,
                    self.linewidth/2
                )
        else:
            op_name = self.trim_op_name(node)
            for node_index in node_indices:
                half_x_width = self.font_size * len(op_name) / 2 / 2 + 1
                half_y_width = self.opheight / 2
                op_y = node_index*self.linespacing + self.opheight
                poly_str = '<rect x="{}" y="{}" width="{}" height="{}" style="fill:white;stroke:black;stroke-width:{}"></rect>'.format(
                    x - half_x_width,
                    op_y - half_y_width,
                    2 * half_x_width,
                    2 * half_y_width,
                    self.opoutlinewidth
                )
                text_str = '<text x="{}" y="{}" font-size="{}" alignment-baseline="middle" text-anchor="middle">{}</text>\n'.format(
                    x, op_y, self.font_size, op_name
                )

                s += poly_str + text_str
        return s

    def feed(self, qbitindex, node):
        # Get new node position
        nodestr = self.trim_op_name(node)

        if nodestr in SvgFeeder.BLACKLIST:
            return

        stringn = len(nodestr)

        total_op_size = self.font_size * stringn / 2 + 1
        new_x_center = self.last_x_max + self.opbuffer + int(total_op_size/2.0)
        new_x_max = new_x_center + total_op_size - int(total_op_size/2.0)

        op_str = self.get_op_string(qbitindex, node, new_x_center)
        self.ops_to_draw.append(op_str)

        # Get ready for next node.
        self.last_x_center = new_x_center
        self.last_x_max = new_x_max

    def build(self):
        acc_str = ''

        # Start by extending lines to new position.
        for i in range(self.n):
            line_y = i*self.linespacing + self.opheight
            line_str = '<line x1="{}" x2="{}" y1="{}" y2="{}" style="stroke:black;stroke-width:{}"></line>\n'.format(
                0, self.last_x_max + self.opbuffer,
                line_y, line_y,
                self.linewidth
            )
            self.svg_acc += line_str

        acc_str += "\n".join(self.ops_to_draw)

        return acc_str

    def get_svg_text(self):
        op_str = self.build()
        tmp = '<svg viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg">\n{}\n{}\n</svg>'.format(
            self.last_x_max + self.opbuffer, self.linespacing*self.n + 2*self.opheight,
            self.svg_acc, op_str
        )
        return tmp


def make_svg(*args, filename=None):
    frontier, graphnodes = get_deps(*args)
    frontier = list(sorted(frontier, key=lambda q: q.qid))
    n = sum(f.n for f in frontier)
    graphacc = SvgFeeder(n)
    run_graph(frontier, graphnodes, graphacc)
    if filename is None:
        return graphacc.get_svg_text()
    else:
        with open(filename, 'w') as w:
            w.write(graphacc.get_svg_text())