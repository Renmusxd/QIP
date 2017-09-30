import numpy


def run(*args, **kwargs):
    feed = kwargs['feed'] if 'feed' in kwargs else {}
    frontier, graphnodes = get_deps(*args, feed=feed)

    for node in frontier:
        if node not in feed:
            raise Exception("Required input not found: {}".format(node))

    feed = feed_forward(frontier, feed, graphnodes)

    if len(args) == 1:
        return feed[args[0]]
    return tuple(feed[arg] for arg in args)

def get_deps(*args, feed=None):
    if feed is None:
        feed = {}

    seen = set(feed.keys())
    deps = set()
    frontier = set(args)
    # Populate frontier
    while len(frontier) > 0:
        node = frontier.pop()
        if node in seen or len(node.inputs) == 0:
            deps.add(node)
        else:
            frontier.update(node.inputs)
        seen.add(node)
    return list(deps), seen

def feed_forward(frontier, feed, graphnodes):
    # Condition is that if a node is in the frontier, either all its inputs are
    # in the feed dict, or it itself is.
    while len(frontier) > 0:
        node = frontier.pop()
        if node not in feed:
            inputval = numpy.concatenate([feed[i] for i in node.inputs])
            feed[node] = node.feed(inputval)
        for nextnode in node.sink:
            if nextnode in graphnodes:
                frontier.append(nextnode)

    return feed
