using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorFlow;

namespace Neuromatic.Optimizers
{
    /// <summary>
    /// Implements the standard stochastic gradient descent algoritm as an optimizer function.
    /// </summary>
    public class SGD : Optimizer
    {
        private double _learningRate;

        /// <summary>
        /// Initializes a new instance of <see cref="SGD"/>
        /// </summary>
        /// <param name="learningRate">Learning rate</param>
        public SGD(double learningRate = 0.01)
        {
            _learningRate = learningRate;
        }

        /// <summary>
        /// Compiles the optimizer
        /// </summary>
        /// <param name="graph">Graph to use for compilation</param>
        /// <param name="loss">Loss function to use</param>
        /// <param name="parameters">Parameters to optimize</param>
        public override void Compile(TFGraph graph, TFOutput loss, IEnumerable<TFOutput> parameters)
        {
            var operations = new List<TFOperation>();

            var learningRate = graph.Const(_learningRate);
            var gradients = graph.AddGradients(new[] { loss }, parameters.ToArray());

            foreach (var (parameter, gradient) in Enumerable.Zip(parameters, gradients, (w, g) => (w, g)))
            {
                var velocity = graph.Mul(gradient, learningRate);
                var newWeight = graph.Add(parameter, velocity);

                var operation = graph.Assign(parameter, newWeight).Operation;

                operations.Add(operation);
            }

            Operations = operations;
        }
    }
}
