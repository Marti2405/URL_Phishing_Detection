import dill
import argparse



def load_model():
    """Load prediction model"""
    # Load the predictor
    with open('url_predictor.dill', 'rb') as f:
        model = dill.load(f)   

    return model

def url_predict(model,urls:list) -> list:
    """
    Input:
        model
        urls:list(strings)
    
    Output:
        phishing_detected:list(float)
    """
    return model.predict(urls)


def main():

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Predict phishing probability for URLs.')
    parser.add_argument('urls', metavar='URL', type=str, nargs='+', help='URLs to classify')

    # Parse the arguments
    args = parser.parse_args()

    # Get the URLs from the arguments
    urls = args.urls

    # Load model
    model = load_model()

    # Predict phishing scores for the provided URLs
    scores = url_predict(model=model,urls=urls)

    # Output the results
    for url, score in zip(urls, scores):
        print(f"URL: {url} | Phishing Probability: {score:.4f}")

if __name__ == "__main__":
    main()


