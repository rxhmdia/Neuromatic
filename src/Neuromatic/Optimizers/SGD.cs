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
        private readonly double _learningRate;
        private readonly double _momentum;

        /// <summary>
        /// Initializes a new instance of <see cref="SGD"/>
        /// </summary>
        /// <param name="learningRate">Learning rate</param>
        /// <param name="momentum">Extra parameter to control how to overcome local optimi</param>
        public SGD(double learningRate = 0.01, double momentum = 0.0)
        {
            _learningRate = learningRate;
            _momentum = momentum;
        }

        /// <summary>
        /// Compiles the optimizer
        /// </summary>
        /// <param name="graph">Graph to use for compilation</param>
        /// <param name="loss">Loss function to use</param>
        /// <param name="parameters">Parameters to optimize</param>
        public override void Compile(ModelCompilationContext context, TFOutput loss, IEnumerable<TFOutput> parameters)
        {
            var graph = context.Graph;
            var operations = new List<TFOperation>();

            var moments = new List<TFOutput>();

            foreach (var parameter in parameters)
            {
                var moment = graph.VariableV2(graph.GetTensorShape(parameter), TFDataType.Double);
                var initializer = graph.Zeros(graph.GetTensorShape(parameter)).Operation;

                context.AddInitializers(initializer);

                moments.Add(moment);
            }

            var momentum = graph.Const(_momentum);

            var learningRate = graph.Const(_learningRate);
            var gradients = graph.AddGradients(new[] { loss }, parameters.ToArray());

            foreach (var (parameter, gradient, moment) in ZipLearningParameters(parameters, gradients, moments))
            {
                // velocity = momentum * moment - learningRate * gradient
                var velocity = graph.Sub(graph.Mul(momentum, moment), graph.Mul(gradient, learningRate));
                var newWeight = graph.Add(parameter, velocity);

                var operation = graph.Assign(parameter, newWeight).Operation;

                operations.Add(operation);
            }

            Operations = operations;
        }

        private IEnumerable<(TFOutput weight, TFOutput gradient, TFOutput momentum)> ZipLearningParameters(
            IEnumerable<TFOutput> parameters, 
            IEnumerable<TFOutput> gradients, 
            IEnumerable<TFOutput> moments)
        {
            var zippedParamsWithGradients = Enumerable.Zip(parameters, gradients, (w, g) => (w, g));

            return Enumerable.Zip(zippedParamsWithGradients, moments, (wg, m) => (wg.w, wg.g, m));
        }
    }
}
