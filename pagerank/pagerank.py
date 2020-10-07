import os
import random
import re
import sys
import numpy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    
    # Initialise variables, lists and dicts
    pageLinks = len(corpus[page])
    totalLinks = len(corpus)
    pageLinksList = []
    output = {}

    # Loop over corpus[page] and append page links to a list
    for link in corpus[page]:
        pageLinksList.append(link)
    
    # To prevent dividing by 0, check the number of links on the page is above 0
    if pageLinks > 0:
        # Calculate probablity of going to any link using 1-damping factor
        allProb = (1 - damping_factor) / totalLinks
        # Calculate probablity of going to a link on the page with damping factor
        pageProb = (damping_factor + (pageLinks * allProb)) / pageLinks
    else:
        # Calculate probablity of going to any link if no outgoing links
        allProb = 1 / totalLinks
        pageProb = 0
    
    # Loop over corpus keys and fill out output dictionary with page and probability
    for key in corpus:
        if key in pageLinksList:
            output[key] = pageProb
        else:
            output[key] = allProb
 
    # Return output
    return output


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initalise a dictionary to store the count of each page that comes up in the sampling
    pages = {}
    for key in corpus:
        pages[key] = 0
    
    # Choose the first page at random
    samplePage = []
    samplePage.append(random.choice(list(pages)))

    # Intialise variable to store transitional_model and do this for first sample 
    tm = transition_model(corpus, samplePage[0], damping_factor)
    
    # Initialise variables to store just page names in a list and the probablities for that page in a list
    pageNames = []
    pageProbs = []

    # Loop until n samples gathered
    for i in range(n):
        pages[samplePage[0]] += 1
        pageNames.clear()
        pageProbs.clear()
        for page, prob in tm.items():
            pageNames.append(page)
            pageProbs.append(prob)
        samplePage = numpy.random.choice(pageNames, 1, p=pageProbs)
        tm = transition_model(corpus, samplePage[0], damping_factor)
    
    # Initialise output dictionary to store PageRank for each page
    output = {}

    for key in pages:
        prob = pages[key] / n
        output[key] = prob
   
    # Return output
    return output


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Total number of pages
    totalPages = len(corpus)

    # Initialise a dictionary to store how many links each page has
    linksFromPage = {}
    
    # Initiaise a dictionary to store PageRanks for each page in the corpus
    pageRanks = {}

    # Initialised a dictionary to store new pageRanks for each page
    newPR = {}

    # Initialise a dictionary to store the difference between current and previous PageRanks
    diff = {}

    # Initialise a list to store sinks
    sinks = []
    
    # Loop through corpus and fill initial values
    for key in corpus:
        pageRanks[key] = 1 / totalPages
        diff[key] = 0
        linksFromPage[key] = len(corpus[key])
        if linksFromPage[key] == 0:
            sinks.append(key)
    
    # Loop through corpus and calculate new PageRanks
    # Only break out of the while loop when PageRank values converge
    converged = False
    while (converged == False):
        # Caluclate New PageRanks based on old PageRanks
        # PR(p) = 1-d/n + d*SUM(PR(i)/out(i)) + d*SUM(PR(x)/n)
        # d = damping factor, n = total number of pages
        # i = pages that link to PR(p)
        # out(i)  =  number of outbound links from pi
        # x = sink nodes, pages with no outbound links
        for page in corpus:
            # Calculate 1-d/n
            allPageProb = (1 - damping_factor) / totalPages
            # Find all the pages that link to PR(p)
            linksToPage = []
            for key in corpus:
                if page in corpus[key]:
                    linksToPage.append(key)
            # Loop through i pages (pages that link to PR(p)) to calculate SUM(PR(i)/out(i))
            eSum = 0.0
            for link in linksToPage:
                eSum += (pageRanks[link] / linksFromPage[link])
            # Loop through x pages (sink nodes) to calculate SUM(PR(x)/n)
            sinkSum = 0.0
            for s in sinks:
                sinkSum += pageRanks[s] / totalPages
            # Multiply the damping factor with each sums
            eSum *= damping_factor
            sinkSum *= damping_factor
            # Calculate the new PR(p)
            newPR[page] = allPageProb + eSum + sinkSum
            # Calculate the new difference between old and new PR(p)
            diff[page] = abs((pageRanks[page] - newPR[page]))
        
        # Update PageRanks dictionary
        for pg in pageRanks:
            pageRanks[pg] = newPR[pg]
        
        # Check the difference between old PageRank and new PageRank
        diff_check = 0
        for page in diff:
            if diff[page] <= 0.001:
                diff_check += 1
        if diff_check == totalPages:
            converged = True
    
    # Return the final PageRanks for each page
    return pageRanks


if __name__ == "__main__":
    main()
