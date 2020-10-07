import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def find_gene_copies(name, one_gene, two_genes):
    """
    Returns the number of copies of the gene that a person has.
    Returns 0, 1 or 2.
    """
    if name in one_gene:
        return 1
    if name in two_genes:
        return 2
    return 0


def find_trait(name, have_trait):
    """
    Return true if person has trait and false if not
    """
    if name in have_trait:
        return True
    return False


def prob_child_gene(child_gene, mum_gene, dad_gene):
    """
    Returns the probablity of a child having a certain number of copies of the gene.
    It takes into account the number of copies each parent has.
    """
    # The probabilities of passing on gene by parent depending on the number of copies they have
    getting_from_parent = {
        0: {'yes': 0.01, 'no': 0.99},
        1: {'yes': 0.5, 'no': 0.5},
        2: {'yes': 0.99, 'no': 0.01}
    }
    # Compute the probability of child getting a certain number of copies depending on parent copies
    if child_gene == 0:
        return getting_from_parent[mum_gene]['no'] * getting_from_parent[dad_gene]['no']
    elif child_gene == 1:
        return (
            (getting_from_parent[mum_gene]['yes'] * getting_from_parent[dad_gene]['no']) + 
            (getting_from_parent[dad_gene]['yes'] * getting_from_parent[mum_gene]['no'])
        )
    else:
        return getting_from_parent[mum_gene]['yes'] * getting_from_parent[dad_gene]['yes']


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """  
    # Initialise a dictionary to store the probabilities
    probs = {}

    # Loop over people and calculate everyone's individual probabilities
    for p in people:
        # Check if child
        if people[p]['mother'] and people[p]['father']:
            # Store the number of copies of gene for each person: child, mother and father
            child_gene = find_gene_copies(p, one_gene, two_genes)
            m_gene = find_gene_copies(people[p]['mother'], one_gene, two_genes)
            f_gene = find_gene_copies(people[p]['father'], one_gene, two_genes)
            
            # Use those values to compute the probablity of child getting that number of copies
            get_gene_prob = prob_child_gene(child_gene, m_gene, f_gene)
            
            # Use PROBS to find the probablity of having/not having trait depending on copies of gene
            prob_trait = PROBS["trait"][child_gene][find_trait(p, have_trait)]
            
            # Calculate overall probablity of have/not having gene with that number of copies of the gene
            probs[p] = get_gene_prob * prob_trait
        
        # Else it is a parent or person with not parent data so only use PROBS data
        else:
            # Find the copies of gene this person has
            gene_copies = find_gene_copies(p, one_gene, two_genes)

            # Use the PROBS values to find probablity of having that number of copies
            gene_prob = PROBS['gene'][gene_copies]

            # Use PROBS to find the probablity of having/not having trait depending on copies of gene
            prob_trait = PROBS["trait"][gene_copies][find_trait(p, have_trait)]
            
            # Calculate overall probablity of have/not having gene with that number of copies of the gene
            probs[p] = gene_prob * prob_trait
    
    # Initialise variable to store joint probability
    jp = 1

    # Loop over the probs dictionary to multiply all values together
    for p in probs:
        jp *= probs[p]
    
    # Return joint probability
    return jp


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # Loop through probabilities and update
    # Function find_gene_copies is used to find the number of copies that person has
    # Function find_trait is used to find if person has trait or not, returns true or false
    for person in probabilities:
        num = find_gene_copies(person, one_gene, two_genes)
        probabilities[person]['gene'][num] += p
        b = find_trait(person, have_trait)
        probabilities[person]['trait'][b] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # Loop through probabilities and normalize
    for person in probabilities:
        # Add all the gene probablities
        gene_sum = sum(probabilities[person]['gene'].values())
        if gene_sum > 0:
            # Work out the factor to multiply by to get all gene probs to add up to 1
            gene_factor = 1 / gene_sum
            # Loop through gene probs and multiply with gene_factor
            for num in probabilities[person]['gene']:
                probabilities[person]['gene'][num] *= gene_factor
        # Add all the trait probs up
        trait_sum = sum(probabilities[person]['trait'].values())
        if trait_sum > 0:
            # Work out the factor to multiply by to get all trait probs to add up to 1
            trait_factor = 1 / trait_sum
            # Loop through trait probs and multiply with trait_factor
            for b in probabilities[person]['trait']:
                probabilities[person]['trait'][b] *= trait_factor


if __name__ == "__main__":
    main()
