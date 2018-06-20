using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic
{
    /// <summary>
    /// A model is defined as a set of inputs and a set of outputs connected to the provided inputs.
    /// When you define a model you also define a training configuration used to optimize the model.
    /// </summary>
    public class Model
    {
        private IEnumerable<Input> _inputs;
        private IEnumerable<Layer> _outputs;
        private TrainingConfiguration _trainingConfiguration;
    }
}
