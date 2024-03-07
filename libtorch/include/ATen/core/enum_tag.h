#pragma once

// @generated by torchgen/gen.py from enum_tag.h

namespace at {
    // Enum of valid tags obtained from the entries in tags.yaml
    enum class Tag {
        core,
        data_dependent_output,
        dynamic_output_shape,
        generated,
        inplace_view,
        nondeterministic_bitwise,
        nondeterministic_seeded,
        pointwise,
        pt2_compliant_tag,
        view_copy
    };
}