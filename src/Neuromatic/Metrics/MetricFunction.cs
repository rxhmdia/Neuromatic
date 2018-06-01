using Neuromatic.Core;
using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.Metrics
{
    /// <summary>
    /// Defines a function that measures a performance metric for a model
    /// </summary>
    public abstract class MetricFunction
    {
        /// <summary>
        /// Compiles the metric function
        /// </summary>
        /// <param name="output"></param>
        /// <param name="backend"></param>
        public abstract void Compile(Layer output, ModelBackend backend);
    }
}
