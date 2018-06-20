using Neuromatic.Losses;
using System;
using System.Collections.Generic;

namespace Neuromatic
{
    /// <summary>
    /// A training configuration defines how a model should be optimized.
    /// This configuration defines the loss functions for all model outputs
    /// and an optimizer to use when training the model with sample data.
    /// </summary>
    public class TrainingConfiguration
    {
        private IEnumerable<LossFunction> _losses;
        private Optimizer _optimizer;

        /// <summary>
        /// Initializes a new instance of <see cref="TrainingConfiguration"/>
        /// </summary>
        /// <param name="losses">A set of loss function to determine the cost of each output defined in the model</param>
        /// <param name="optimizer">The optimization function used to optimize the model</param>
        public TrainingConfiguration(IEnumerable<LossFunction> losses, Optimizer optimizer)
        {
            _losses = losses;
            _optimizer = optimizer;
        }


    }
}
