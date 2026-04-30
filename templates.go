package main

import (
	"embed"
	"html/template"
	"strconv"
	"strings"
)

//go:embed templates/*.tmpl templates/partials/*.tmpl
var templateFS embed.FS

func parseTemplates() (*template.Template, error) {
	funcs := template.FuncMap{
		"replaceAll": strings.ReplaceAll,
		"hasPrefix":  strings.HasPrefix,
		"capitalize": capitalize,
		"commaInt": func(n int) string {
			s := strconv.Itoa(n)
			if n < 1000 {
				return s
			}
			var out []byte
			pre := len(s) % 3
			if pre == 0 {
				pre = 3
			}
			out = append(out, s[:pre]...)
			for i := pre; i < len(s); i += 3 {
				out = append(out, ',')
				out = append(out, s[i:i+3]...)
			}
			return string(out)
		},
		"slice": func(items ...string) []string {
			return items
		},
		"js": mustJSONJS,
	}
	return template.New("root").Funcs(funcs).ParseFS(templateFS, "templates/*.tmpl", "templates/partials/*.tmpl")
}

func capitalize(s string) string {
	if s == "" {
		return s
	}
	if len(s) == 1 {
		return strings.ToUpper(s)
	}
	return strings.ToUpper(s[:1]) + s[1:]
}
