"""
Evaluation Script for Quantsphere Trading Agent

This script evaluates the trained Quantsphere trading agent on test data.
"""

import os
import sys
import logging
import argparse
from typing import List, Tuple, Optional
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantsphere import TradingAgent
from quantsphere.utils import (
    get_stock_data, 
    show_eval_result, 
    switch_k_backend_device,
    setup_logging,
    create_trading_visualization,
    calculate_metrics,
    save_results
)


def evaluate_agent(
    test_data_path: str,
    window_size: int = 10,
    model_name: Optional[str] = None,
    debug: bool = False,
    save_plot: bool = False,
    save_results_file: bool = False
) -> None:
    """
    Evaluate the Quantsphere trading agent
    
    Args:
        test_data_path: Path to test data CSV
        window_size: State window size
        model_name: Model name to evaluate (None for all models)
        debug: Whether to use debug logging
        save_plot: Whether to save trading visualization
        save_results_file: Whether to save results to file
    """
    # Setup logging
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(log_level)
    
    # Switch to CPU if needed
    switch_k_backend_device()
    
    # Load test data
    logging.info(f"Loading test data from {test_data_path}")
    test_data = get_stock_data(test_data_path)
    initial_offset = test_data[1] - test_data[0] if len(test_data) > 1 else 0
    
    # Get list of models to evaluate
    if model_name:
        models_to_eval = [model_name]
    else:
        models_to_eval = []
        if os.path.exists("models"):
            for file in os.listdir("models"):
                if os.path.isfile(os.path.join("models", file)):
                    models_to_eval.append(file)
    
    if not models_to_eval:
        logging.warning("No models found to evaluate")
        return
    
    # Evaluate each model
    for model in models_to_eval:
        try:
            logging.info(f"Evaluating model: {model}")
            
            # Initialize agent with pretrained model
            agent = TradingAgent(
                state_size=window_size,
                pretrained=True,
                model_name=model
            )
            
            # Evaluate agent
            profit, history = agent.evaluate(test_data, debug=debug)
            
            # Show results
            show_eval_result(model, profit, initial_offset)
            
            # Calculate detailed metrics
            metrics = calculate_metrics(history)
            logging.info(f"Detailed metrics for {model}:")
            logging.info(f"  Total Return: {metrics['total_return']:.2f}%")
            logging.info(f"  Total Trades: {metrics['total_trades']}")
            logging.info(f"  Win Rate: {metrics['win_rate']:.1f}%")
            logging.info(f"  Average Profit: ${metrics['avg_profit']:.2f}")
            logging.info(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
            
            # Save visualization if requested
            if save_plot:
                fig = create_trading_visualization(
                    test_data, 
                    history, 
                    title=f"Trading Performance - {model}"
                )
                plot_filename = f"trading_plot_{model}.png"
                fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
                logging.info(f"Trading visualization saved as {plot_filename}")
            
            # Save results if requested
            if save_results_file:
                results = {
                    'model_name': model,
                    'test_data': test_data_path,
                    'total_profit': profit,
                    'initial_offset': initial_offset,
                    **metrics
                }
                results_filename = f"results_{model}.csv"
                save_results(results, results_filename)
                logging.info(f"Results saved as {results_filename}")
            
            # Clean up agent to free memory
            del agent
            
        except Exception as e:
            logging.error(f"Error evaluating model {model}: {e}")
            continue
    
    logging.info("Evaluation completed!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Evaluate Quantsphere Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py data/GOOG_2019.csv --model-name model_GOOG_50 --debug
  python eval.py data/AAPL_2019.csv --save-plot --save-results
  python eval.py data/TSLA_2019.csv --window-size 20
        """
    )
    
    parser.add_argument("test_data", help="Path to test data CSV file")
    parser.add_argument(
        "--window-size", 
        type=int, 
        default=10,
        help="Size of the state window (default: 10)"
    )
    parser.add_argument(
        "--model-name", 
        help="Name of the model to evaluate (evaluates all models if not specified)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--save-plot", 
        action="store_true",
        help="Save trading visualization plot"
    )
    parser.add_argument(
        "--save-results", 
        action="store_true",
        help="Save detailed results to CSV file"
    )
    
    args = parser.parse_args()
    
    try:
        evaluate_agent(
            test_data_path=args.test_data,
            window_size=args.window_size,
            model_name=args.model_name,
            debug=args.debug,
            save_plot=args.save_plot,
            save_results_file=args.save_results
        )
    except KeyboardInterrupt:
        logging.info("Evaluation interrupted by user")
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
