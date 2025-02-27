mindspore.ops.conv3d
====================

.. py:function:: mindspore.ops.conv3d(input, weight, bias=None, stride=1, pad_mode="valid", padding=0, dilation=1, groups=1)

    对输入Tensor计算三维卷积。该Tensor的常见shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` ，其中 :math:`N` 为batch size，:math:`C_{in}` 为通道数，:math:`D` 为深度， :math:`H_{in}, W_{in}` 分别为特征层的高度和宽度。 :math:`X_i` 为 :math:`i^{th}` 输入值， :math:`b_i` 为 :math:`i^{th}` 输入值的偏置项。对于每个batch中的Tensor，其shape为 :math:`(C_{in}, D_{in}, H_{in}, W_{in})` ，公式定义如下：

    .. math::
        \operatorname{out}\left(N_{i}, C_{\text {out}_j}\right)=\operatorname{bias}\left(C_{\text {out}_j}\right)+
        \sum_{k=0}^{C_{in}-1} ccor(\text {weight}\left(C_{\text {out}_j}, k\right),
        \operatorname{input}\left(N_{i}, k\right))

    其中，:math:`k` 为卷积核数，:math:`ccor` 为 `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ ，
    :math:`C_{in}` 为输入通道数， :math:`j` 的范围从 :math:`0` 到 :math:`C_{out} - 1` ， :math:`W_{ij}` 对应第 :math:`j` 个过滤器的第 :math:`i` 个通道， :math:`out_{j}` 对应输出的第 :math:`j` 个通道。
    :math:`W_{ij}` 为卷积核的切片，其shape为 :math:`(\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` ，其中 :math:`\text{kernel_size[1]}` 和 :math:`\text{kernel_size[2]}` 是卷积核的高度和宽度，
    :math:`\text{kernel_size[0]}` 是卷积核的深度。
    :math:`\text{bias}` 是偏置参数， :math:`\text{X}` 是输入Tensor。
    完整卷积核的shape为 :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` ，其中 `group` 是在通道上分割输入 `inputs` 的组数。


    详细内容请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。

    .. note::

        1. 在Ascend平台上，目前只支持深度卷积场景下的分组卷积运算。也就是说，当 `groups>1` 的场景下，必须要满足 :math:`C_{in} = C_{out} = groups` 的约束条件。
        2. 在Ascend平台上，目前只支持 :math:`dialtion=1` 。

    参数：
        - **input** (Tensor) - shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。
        - **weight** (Tensor) - shape为 :math:`(C_{out}, C_{in} / \text{groups}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`  ，则卷积核的大小为 :math:`(\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})` 。
        - **bias** (Tensor) - 偏置Tensor，shape为 :math:`(C_{out})` 的Tensor。如果 `bias` 是None，将不会添加偏置。默认值：None。
        - **stride** (Union[int, tuple[int]]，可选) - 卷积核移动的步长，数据类型为int或两个int组成的tuple。一个int表示在高度和宽度方向的移动步长均为该值。两个int组成的tuple分别表示在高度和宽度方向的移动步长。默认值：1。
        - **pad_mode** (str，可选) - 指定填充模式。取值为"same"，"valid"，或"pad"。默认值："valid"。

          - **same**: 输出的高度和宽度分别与输入整除 `stride` 后的值相同。填充将被均匀地添加到高和宽的两侧，剩余填充量将被添加到维度末端。若设置该模式，`padding` 的值必须为0。
          - **valid**: 在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。如果设置此模式，则 `padding` 的值必须为0。
          - **pad**: 对输入 `input` 进行填充。在输入的高度和宽度方向上填充 `padding` 大小的0。如果设置此模式， `padding` 必须大于或等于0。
        
        - **padding** (Union[int, tuple[int]]，可选) - 输入 `input` 的深度、高度和宽度方向上填充的数量。数据类型为int或包含3个int组成的tuple。如果 `padding` 是一个int，那么前、后、上、下、左、右的填充都等于 `padding` 。如果 `padding` 是一个有3个int组成的tuple，那么前、后的填充为 `padding[0]` ，上、下的填充为 `padding[1]` ，左、右的填充为 `padding[2]` 。值必须大于等于0，默认值：0。
        - **dilation** (Union[int, tuple[int]]，可选) - 卷积核元素间的间隔。数据类型为int或由3个int组成的tuple。若 :math:`k > 1` ，则卷积核间隔 `k` 个元素进行采样。前后、垂直和水平方向上的 `k` ，其取值范围分别为[1, D]、[1, H]和[1, W]。默认值：1。
        - **groups** (int，可选) - 将筛选器拆分为组，默认值：1。当前仅支持值为1。

    返回：
        Tensor，卷积后的值。shape为 :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` 。

        `pad_mode` 为"same"时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lceil{\frac{D_{in}}{\text{stride[0]}}} \right \rceil \\
                H_{out} = \left \lceil{\frac{H_{in}}{\text{stride[1]}}} \right \rceil \\
                W_{out} = \left \lceil{\frac{W_{in}}{\text{stride[2]}}} \right \rceil \\
            \end{array}

        `pad_mode` 为"valid"时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) }
                {\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) }
                {\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} - \text{dilation[2]} \times (\text{kernel_size[2]} - 1) }
                {\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

        `pad_mode` 为"pad"时：

        .. math::
            \begin{array}{ll} \\
                D_{out} = \left \lfloor{\frac{D_{in} + padding[0] + padding[0] - (\text{dilation[0]} - 1) \times
                \text{kernel_size[0]} - 1 }{\text{stride[0]}} + 1} \right \rfloor \\
                H_{out} = \left \lfloor{\frac{H_{in} + padding[1] + padding[1] - (\text{dilation[1]} - 1) \times
                \text{kernel_size[1]} - 1 }{\text{stride[1]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} + padding[2] + padding[2] - (\text{dilation[2]} - 1) \times
                \text{kernel_size[2]} - 1 }{\text{stride[2]}} + 1} \right \rfloor \\
            \end{array}

    异常：
        - **TypeError** -  `stride` 、 `padding` 或 `dilation` 既不是int也不是tuple。
        - **TypeError** -  `groups` 不是int。
        - **TypeError** -  `bias` 不是Tensor。
        - **ValueError** - `bias` 的shape不是 :math:`(C_{out})` 。
        - **ValueError** - `stride` 或 `diation` 小于1。
        - **ValueError** - `pad_mode` 不是"same"、"valid"或"pad"。
        - **ValueError** - `padding` 是一个长度不等于3的tuple。
        - **ValueError** - `pad_mode` 不等于"pad"时，`padding` 大于0。
