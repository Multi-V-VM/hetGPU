/* Aggregates all XRT C headers bindgen should wrap.
 * Only C-API headers (with extern "C" sections) are included.
 * xrt_hw_context.h is C++-only (#error under C parser).
 * xrt_uuid.h is transitively included via xrt_device.h.
 */
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
